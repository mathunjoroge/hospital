<?php 
require_once('../main/auth.php');
include('../connect.php');

if (!isset($_GET['response'])) {
   $_GET['response']=0;
 } 
?>
<!DOCTYPE html>
<html>
<title>services per specialist</title>

<?php
include "../header.php";
?>
</head>
<header class="header clearfix" style="background-color: #3786d6;;">
<?php include('../main/nav.php'); ?>   
</header><?php include('sidee.php'); ?>
<div class="content-wrapper">
<script>
$( function() {
$( "#mydate" ).datepicker({
changeMonth: true,
changeYear: true
});
} );

</script>
<script>
$( function() {
$( "#mydat" ).datepicker({
changeMonth: true,
changeYear: true
});
} );
</script>  
<link rel="stylesheet" href="../main/jquery-ui.css">
<script src="../main/jquery-1.12.4.js"></script>
<script src="../main/jquery-ui.js"></script>
<div class="jumbotron" style="background: #95CAFC;">
<nav>
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">services by specialists</li>
<li class="breadcrumb-item active" aria-current="page"><?php echo $_GET["name"]; ?></li>
</li>
</ol>
</nav>
<div class="container" align="center">         
<form action="report.php" method="GET">
<input type="hidden" name="id" value="<?php echo $_GET['id'] ?>">
<input type="hidden" name="name" value="<?php echo $_GET['name'] ?>">
from: <input type="text" id="mydate"  name="d1" autocomplete="off" placeholder="pick start date" required/> to: <input type="text" id="mydat"  name="d2" autocomplete="off" placeholder="pick end date" required/>
<button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>     
</div>
</form>
<?php 
if (isset($_GET["d1"])) {
# code...

?>
<div class="container">
<?php
$d1=$_GET['d1']." 00:00:00"; 
$d2=$_GET['d2']."".date("H:i:s");
$date1=date("Y-m-d H:i:s", strtotime($d1));
$date2=date("Y-m-d H:i:s", strtotime($d2)); 
?>
<p>&nbsp;</p>
<p>services offered by <?php echo $_GET['name']; ?>  from <?php echo date("d-m-Y", strtotime($date1)); ?> to <?php echo date("d-m-Y", strtotime($date1)); ?> </p>


<table class="table table-bordered">
<thead>
<tr>
<th>date</th>
<th>patient</th>
<th>description</th>
<th>amount</th>
<th>action</th>

</tr>
</thead>
<tbody>
<?php
$done_by= $_GET['id'];
$paid_out=0;
$result = $db->prepare("SELECT fees.fees_name AS fees,patients.name AS patient_name,collection.amount AS amount, collection.date as date,collection_id  FROM collection
JOIN fees ON collection.fees_id=fees.fees_id 
JOIN patients ON collection.paid_by=patients.opno 
WHERE (date(collection.date) >=:a AND date(collection.date)<= :b) AND done_by=:done_by AND paid_out=:paid_out");
$result->bindParam(':a',$date1);
$result->bindParam(':b',$date2);
$result->bindParam(':done_by',$done_by);
$result->bindParam(':paid_out',$paid_out); 
  
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
?>

<tr>
<td><?php echo date("d-m-Y", strtotime($row["date"])); ?></td>
<td><?php echo $row["patient_name"]; ?></td>
<td><?php echo $row["fees"]; ?></td>
<td><?php echo $row["amount"]; ?></td>
<form id="form"  action="update.php" method="POST">
    <td>
    <?php $id=$row["collection_id"]; ?>
     <input type="checkbox" name="id[]" id="id" title="click to pay" value="<?php echo $id; ?>">
    </td>

</tr>
<?php } ?>

</form>
</tbody>
</table>
<?php } ?>
<?php
$result = $db->prepare("SELECT sum(amount) AS total, collection.date as date FROM  collection
WHERE (date(collection.date) >=:a AND date(collection.date)<= :b) AND done_by=:done_by AND paid_out=:paid_out");
$result->bindParam(':a',$date1);
$result->bindParam(':b',$date2);
$result->bindParam(':done_by',$done_by);
$result->bindParam(':paid_out',$paid_out); 
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
?>
<table class="table table-bordered" style="width:90%;border: none;">
  <thead>
    <tr>
      <th style="border: none;">total</th>
      <th width="70" style="border: none;"><?php echo $row['total']; ?></th><?php } ?>
    </tr>
  </thead>
</table>
</div>
<div><button class="btn btn-success btn-large" style="width: 100%;">save as paid</button></a></div>
</div>
 <script type="text/javascript">  
        $(function(){
         $('.checkbox').on('change',function(){
            $('#form').submit();
            });
        });
    </script>

</body>
</html>