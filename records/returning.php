<?php 
require_once('../main/auth.php');
?>
<!DOCTYPE html>
<html>
<title>returning patient</title>
<?php 
include('../header.php');
?>
</head>
  <header class="header clearfix" style="background-color: #3786d6;;">
    

    </button>
    <?php include('../main/nav.php'); ?>   
  </header><?php include('side.php'); ?>
  <div class="content-wrapper">    
      <div class="jumbotron" style="background: #95CAFC;">         
   <script>
  $( function() {
    $( "#mydate" ).datepicker({
      yearRange: "-0:+10",
      changeMonth: true,
      changeYear: true

    });
  } );

  </script>
  
</head><div>
     <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
    <li class="breadcrumb-item active" aria-current="page">returning patient</li>
    <?php
    if (isset($_REQUEST["search"])) {
      # code...
    
      $search=$_GET['search'];
      include ('../connect.php');
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
        $result->execute();
        for($i=0; $row = $result->fetch(); $i++){
        $a=$row['name'];
        $b=$row['age'];
        $c=$row['sex'];
        $d=$row['opno'];
     
     ?>
     <li class="breadcrumb-item active" aria-current="page"><?php echo $a; ?> age: <?php include '../doctors/age.php'; ?> &nbsp; <?php echo $c; ?></li><?php }} ?>
     </ol>
</nav> 

<form method="GET" action="returning.php" >
  <span><?php 
include('../pharmacy/patient_search.php');
?>

  <input type="hidden" name="response" value="0"> <button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>     
</div></form>
<?php if (isset($_REQUEST["search"])) {
  # code...
 ?>
&nbsp;
<p>&nbsp;</p>
<p>&nbsp;</p>
<div class="container">
  <form method="POST" action="save_returning.php">
    <input type="hidden" name="patient" value="<?php echo $_REQUEST["search"]; ?>">
    <input type="hidden" name="name" value="<?php echo $a; ?>">
    <div class="row">
    <div class="form-group col-md-6">
        <label for="inputPassword4">select fees to be paid for</label></br>
    <select id="fee" class="selectpicker form-control" title="select fees, mulptiple allowed" data-live-search="true" name="fees[]" class="form-control" style="width: 70%;" multiple  required/>
    <option value=""><?php 
 include ('../connect.php');
$result = $db->prepare("SELECT * FROM fees");
        $result->execute();
        for($i=0; $row = $result->fetch(); $i++){
           echo "<option value=".$row['fees_id'].">".$row['fees_name']."</option>";
         }
        ?>      
</select>
</div>
<div class="form-group col-md-6">
<label for="inputPassword4">select lab tests to be paid</label></br>
<select id="lab" class="selectpicker form-control"  title="select lab tests, mulptiple allowed"  name="lab[]"  multiple/>
<option value="">-- Select payable fees--</option><?php 
include ('../connect.php');
$result = $db->prepare("SELECT * FROM lab_tests");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
echo "<option value=".$row['id'].">".$row['name']."</option>";
}

?>      
</select>
</div>
</div>
<button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button>
    
  </form>
</div>
<?php } ?>
</div></div></div>

<?php
if ($_GET['response']==1) {
  # code...


 ?>
<div class="alert alert-success"  style="width: 21%;margin-left: 20%;"> patient data successifuly</div>
<?php } ?>
<?php 
if ($_GET['response']==0) {
  
}
 ?>

</body>
</html>