<?php 
require_once('../main/auth.php');
include ('../connect.php');
$shownav=0; ?>
<!DOCTYPE html>
<html>
<title>test list</title><?php 
include "../header.php";
?>
</head>
<body><header class="header clearfix" style="background-color: #3786d6;">
<button type="button" id="toggleMenu" class="toggle_menu">
<i class="fa fa-bars"></i>
</button>
<?php include('../main/nav.php'); ?>   
</header><?php include('side.php'); ?>
<div class="content-wrapper"> <div class="jumbotron" style="background: #95CAFC;">

<div class="container">
<?php if (isset($_GET['response'])) {
# code...
?>
<p class="alert-success">parameters saved!</p>
<?php } ?>
<table class="resultstable" >
<thead>
<tr>
<th> name</th>
<th> sex</th>
<th>amount</th>
<th>view parameters</th>
<th>multiple param</th>
<th>action</th>
</tr>
</thead>
<?php
$result = $db->prepare("SELECT lab_tests.id AS id,lab_tests.name AS name,lab_tests.cost AS cost,refs_table.sex AS sex,lab_tests.template AS template FROM lab_tests left  JOIN refs_table ON lab_tests.id=refs_table.test_id order by name");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){

$name = $row['name'];
$amount = $row['cost']; 
$template = $row['template'];
$sex = $row['sex'];

?>
<tbody>
<tr>
<td><?php echo $name; ?></td>
<td><?php
if ($sex==1) {
	echo "male ranges";
}
if ($sex==2) {
	echo "female ranges";
}
if ($sex==3) {
	echo "children ranges";
}
if ($sex==4) {
	echo "infant ranges";
}
 ?></td>
<td><?php echo $amount; ?></td>
<td><?php
// check if a template for the lab results is not empty
if (empty($template)){ ?>no set parameters<?php } ?> 
<?php if (!empty($template)) {
# code...
?><a rel="facebox" href="test_details.php?test_id=<?php echo $row['id']; ?>&test_name=<?php echo $name; ?>&sex=<?php echo $sex; ?>">view parameters</a><?php } ?></td>
<td><a href="form.php?id=<?php echo $row['id']; ?>&test=<?php echo $name; ?>">add params</a></td>
<td><a rel="facebox" href="edittest.php?id=<?php echo $row['id']; ?>&name=<?php echo $name; ?>&amount=<?php echo $amount; ?>"><button class="btn btn-success" style="height: 5px;" title="click to edit"></button></a> <button class="btn btn-danger" style="height: 5px;" title="Click to Delete"></button> </td><?php }?>
</tr>
<tr> 
</tbody>
</table>

</div>
</body>
</html>