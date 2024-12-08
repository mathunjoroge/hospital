<?php 
include('../connect.php');
require_once('../main/auth.php');
$d1=date('Y-m-d')." 00:00:00";
$d2=date('Y-m-d H:i:s');
?> 
<!DOCTYPE html>
<html>
<title>patient nursing details</title>
<?php 
include "../header.php"; 
?>
</head>
<body>
<header class="header clearfix" style="background-color: #3786d6;">

<?php include('../main/nav.php'); 
?>   
</header>
<?php include('side.php'); ?>
<div class="content-wrapper">

<div class="jumbotron" style="background: #95CAFC;">
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">patient nursing details</li>

<?php
$search=$_GET['search'];
include ('../connect.php');
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$a=$row['name'];
$b=$row['age'];
$c=$row['sex'];

?>
<li class="breadcrumb-item active" aria-current="page"><?php echo $a; ?></li>
<li class="breadcrumb-item active" aria-current="page"> <?php 
include '../doctors/age.php';
?></li>
<?php } ?>
</ol>
</nav>
<div class="container">
<form action="patient_details.php?" method="GET">
<span><?php
include("../pharmacy/patient_search.php");
?>
<input type="hidden" name="response" value="0"> <button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>     

</form>  

<?php
$search=$_GET['search'];
$nothing="";


if ($search!=$nothing) {
?><?php } ?>
<?php 
$search=$_GET['search'];
$response=0;
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
<div class="container">

<h3>blood presure</h3>
<form action="savepatient.php" method="POST">        
<div class="container">

<table class="table-bordered" id="notes" >
<thead>
<tr>
<th>systolic</th>
<th>diastolic</th>
<th>pulse rate</th>
<th>spo<sub>2</sub></th>
</tr>
</thead>

<tbody>
<tr><td><input class="form-control" type="number"  name="sys" ></td>
<td><input class="form-control" type="number"   name="dys" ></td>
<td><input class="form-control" type="number"   name="rate" ></td>
<td><input class="form-control" type="text"   name="spo" ></td>
</tr>
</tbody>
</table>
</div>
<div class="container">
<h3>physical</h3>

<table class="table-bordered">
<thead>
<tr>
<th>height</th>
<th>weight</th>
<th>temperature</th>
</tr>
</thead>  
<tbody>
<tr>
<td>
<input class="form-control" type="number" class="form-control" step="0.001"   placeholder="cm" name="height" min="46">
</td>
<td>
<input class="form-control" type="number" step="0.001"  placeholder="kgs"   name="weight" min="2" required>
</td>
<td>
<input class="form-control" type="number" step="0.001"  placeholder="degrees c" name="temp" required>
</td>
</tr>
</tbody>
</table>
</div>
<div class="container">
<table class="table-bordered">
<thead>
<tr>
<th>breath rate</th>
<th>rbs</th>
<th>MUAC</th>
</tr>
</thead>

<tbody>
<tr>
<td>
<input class="form-control" type="number" step="0.001"  placeholder="bpm" name="br">
</td>
<td>
<input class="form-control" type="number" step="0.001"  placeholder="mm/L" name="rbs">
<input class="form-control" type="hidden"  placeholder="patient number" name="opno" value="<?php echo $d ?>">
</td>
<td><input class="form-control" type="number" step="0.001"  placeholder="mm" name="muac"></td>
</tr>
</tbody>
</table>

<?php
if ($c=="female") {
# code...
?><div class="container">
<p>&nbsp;</p>
<table class="table-bordered" style="width:auto;">
<head>
<tr>
<th>LMP</th>
<th>EDD</th>
<th>para</th>
<th>gravid</th>
<th>live births</th>
<th>births alive</th>
</tr>
</head>
<tbody>
<tr>
<td><input type="text" id="lmp" name="lmp" autocomplete="off" ></td>
<td><input type="text" id="edd" name="edd" value="" autocomplete="off"></td>
<td><input type="number" name="para" style="width:4em;" value=""></td>
<td><input type="number" name="gravid" style="width:4em;" value=""></td>
<td><input type="number" name="live_births" style="width:4em;" value=""></td>
<td><input type="number" name="births_alive" style="width:4em;" value=""></td>
</tr>
</tbody>
</table>
<?php }  ?>
<p>&nbsp;</p>
<button class="btn btn-success btn-large" style="width: 65%;">save</button></form> </form>
<?php } ?>

</div></div></div>

<script>
$( function() {
$( "#lmp" ).datepicker({
changeMonth: true,
changeYear: true
});
} );
</script>
<script>
$( function() {
$( "#edd" ).datepicker({
changeMonth: true,
changeYear: true
});
} );
</script>
</div>
<script src="dist/vertical-responsive-menu.min.js"></script>

</body>
</html>