<?php 
require_once('../main/auth.php');
include ('../connect.php'); 
if (!isset($_GET['response'])) {
   $_GET['response']=0;
 } 
?>
<!DOCTYPE html>
<html>
<title>professional</title>
<?php
include "../header.php";
?>
</head>
<body>
<script>

$("#table td").each(function (index) {
    $(this).css("background-color", "#"+((1<<24)*Math.random()|0).toString(16));
});</script>
<header class="header clearfix" style="background-color: #3786d6;">
<?php include('../main/nav.php'); ?>   
</header><?php include('sidee.php'); ?>
<div class="jumbotron" style="background: #95CAFC;">
<nav>
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">specialists</li>
</li>

</ol>
</nav>
<div class="container">
<?php if ($_GET['response']==8) {
# code...
?>
<p class="alert alert-warning"> a person by that name exists. please use a different name</p>
<?php } ?>
<?php if ($_GET['response']==9) {
# code...
?>
<p class="alert alert-success"><?php echo $_GET['name']; ?> saved</p>
<?php } ?>


<span><a rel="facebox" href="add_professional.php"> <button class="btn-success" style="">add professional</button></a> 
<table class="table table-bordered table-striped"  style="width:auto;">
<a rel="facebox" href="add_profession.php"> <button class="btn-success" style="">add profession or a job</button></a></span> 
<table class="table table-bordered"  id="table">
    <?php
    if (isset($_GET['success'])) {
    // code...

?>
<script>
setTimeout(function() {
$('#alert').fadeOut('fast');
}, 3000); 
</script>
<p class="text text-success" id="alert">marked as paid</p>
<?php } ?>
<caption>professionals</caption>
<thead class="bg-primary">
<tr>
<th>name</th>
<th>profession</th>
<th>query</th>
</tr>
</thead>
<?php
$result = $db->prepare("SELECT professionals.name AS name, professions.name AS profession, professionals.profession_id AS id  FROM professionals JOIN professions on professionals.profession_id=professions.profession_id");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){     
$name = $row['name'];
$profession = $row['profession'];
?>
<tbody> 
<tr>
<td style="width:auto;"><?php echo $name; ?></td>
<td><?php echo $profession; ?></td>
<td width="10%"><a  href="report.php?id=<?php echo $row['id']; ?>&name=<?php echo $name; ?>"><button class="btn btn-success" style="height: 5px;" title="query report"></button></a></td><?php }?>
</tr>
<tr> 
</tbody>
</table>

</div>
</body>
</html>