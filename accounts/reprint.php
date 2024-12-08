<?php 
require_once('../main/auth.php');
include('../connect.php');
 ?> 
 <!DOCTYPE html>
<html>
<title>total cash</title>

 <?php
 include "../header.php";
  ?>
</head>
  <header class="header clearfix" style="background-color: #3786d6;;">
    <?php include('../main/nav.php'); ?>   
  </header><?php include('side.php'); ?>
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
          <ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page"><b>print receipt copy</b></li>
</ol>
      <div class="container" align="center">
<form action="reprint.php" method="GET">
from: <input type="text" id="mydate"  name="d1" autocomplete="off" placeholder="pick start date" required="true"/> to: <input type="text" id="mydat"  name="d2" autocomplete="off" placeholder="pick end date" required="true"/>
<button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>     
</form>
<?php 
if (isset($_GET["d1"])) {
$d1=$_GET['d1']." 00:00:00"; 
$d2=$_GET['d2']." 23:59:59";
$date1=date("Y-m-d H:i:s", strtotime($d1));
$date2=date("Y-m-d H:i:s", strtotime($d2));
$d=0;
$e=3;

//end of pharmacy table


?>

<center><b>receipts from: <?php echo date("d-m-Y", strtotime($d1)); ?>  to: <?php echo date("d-m-Y", strtotime($d2)); ?> </b></center>
<p></p>

<table class="table">
<tr>
<th>patient name </th>
<th>ip number</th>
<th>receipt number </th>
<th>date</th>
<th>amount</th>
</tr>
<tr>
<?php 
$result = $db->prepare("SELECT name,opno,receipt_no,receipts.date AS date,total,patients.type AS type FROM receipts RIGHT OUTER JOIN patients ON receipts.patient=patients.opno WHERE date(receipts.date)>=:a AND date(receipts.date)<=:b");
$result->bindParam(':a',$date1);
$result->bindParam(':b',$date2);
$result->execute(); 
for($i=0; $row = $result->fetch(); $i++){
$name=$row['name'];
$ip_no=$row['opno'];
$receipt_number=$row['receipt_no'];
$date=$row['date'];
$amount=$row['total'];
$type=$row['type'];

?>
<td><?php echo $name; ?></td>
<td><?php echo $ip_no; ?></td>
<td><a href="receipt_copy.php?receipt=<?php echo $receipt_number; ?>&mode=<?php echo $type; ?>&insurance=<?php  if (isset($company)) { echo $company; 
# code...
} 
else{
echo "0";
}  ?>&search=<?php echo $ip_no; ?>"><?php echo $receipt_number; ?></a></td>
<td><?php echo $date; ?></td>
<td><?php echo $amount; ?></td>
</tr>
<?php } ?>

</table>
<?php } ?>
</div>
</div>
</div>

</body>
</html>