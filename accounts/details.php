<?php
require_once('../main/auth.php');
include('../connect.php');
?>
<!DOCTYPE html>
<html>
<title>request details</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<?php 
include "../header.php";
?>
</head><body >
<header class="header clearfix" style="background-color: #3786d6;;">
<button type="button" id="toggleMenu" class="toggle_menu">
<i class="fa fa-bars"></i>

</button>
<?php
include('../main/nav.php');
?>

</header><?php
include('side.php');
?>
<div class="content-wrapper">  
<div class="jumbotron" style="background: #95CAFC;">
<div class="container"> 
<div class="container"> 
<div>
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">patient</li>
<li class="breadcrumb-item active" aria-current="page">approve tests</li>
<?php
$search = $_GET['search'];
include('../connect.php');
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for ($i = 0; $row = $result->fetch(); $i++) {
$a = $row['name'];
$b = $row['age'];
$c = $row['sex'];
$d = $row['opno'];
?>
<li class="breadcrumb-item active" aria-current="page"><?php
echo $a;
?> <?php include "../doctors/age.php"; ?> &nbsp; </caption></li><?php
}
?>

</nav>  
</div>
<?php
if (isset($search)) {
# code...
$search   = $_GET['search'];
$response = 0;
include('../connect.php');
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o LIMIT 1");
$result->BindParam(':o', $search);
$result->execute();
for ($i = 0; $row = $result->fetch(); $i++) {
$a = $row['name'];
$b = $row['age'];
$c = $row['sex'];
$d = $row['opno'];
?>

<?php
$served = 0;
$patient = $_GET['search'];
$result  = $db->prepare("SELECT name, template, test,opn,reqby,lab_tests.cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn=:a AND lab.served=:b");
$result->bindParam(':a', $patient);
$result->bindParam(':b', $served);
$result->execute();
$rowcountt = $result->rowcount();
//Check whether the query was successful or not
if ($rowcountt >= 1) {
?>
<p>&nbsp;</p>
<div class="container" > 
<label>lab tests requested for</label></br> 
<table class="resultstable" >
<thead>
<tr>
<th>test</th>
<th>requested by</th>
<th>cost</th>
<th>approve</th>
</tr>
</thead>
<?php
$patient = $_GET['search'];
$result  = $db->prepare("SELECT  lab.id AS id,lab_tests.id AS test_id,name, test,opn,reqby,lab_tests.cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn=:a AND lab.served=:b");
$result->bindParam(':a', $patient);
$result->bindParam(':b', $served);
$result->execute();
$result->execute();
for ($i = 0; $row = $result->fetch(); $i++) {
$name     = $row['name'];
$test_id  = $row['test_id'];
$lab_id   = $row['id'];
$reqby    = $row['reqby'];
$cost     = $row['cost'];
$request_id=$lab_id;
?>
<tbody>
<tr>
<td><?php echo $name; ?></td>
<td><?php echo $reqby; ?></td>
<td><?php echo $cost; ?></td>
<td><form action="approve_lab.php" method="POST"><input type="checkbox"  name="lab_id[]" value="<?php echo $lab_id; ?>" recquired/></td>
<?php } ?>
</tr>

</tbody>
</table>
</br>
<button class="btn btn-success btn-large" style="width: 100%;">save</button></a> <?php
}
?></form></div>

<?php
}
?> 
<?php
}
?>
</div></div></div></div>
</div></div>

</body>
</html>