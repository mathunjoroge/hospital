<?php
include('../connect.php');
?>
<form action="submit.php" method="POST">
	<h4 align="center"><?php echo $_GET['name'];  ?></h4>
<table class="resultstable" >
<thead>
<tr>
<th>parameter</th>
<th>normal range</th>
<th>results</th>
</tr>
</thead>
<tbody>
<?php
//lab id is unique id on lab table and will be used to update the template
if (isset($_GET["view"])) {
	$request_id=$_GET["request_id"];
	?><?php
$result = $db->prepare("SELECT*  FROM lab_results RIGHT OUTER JOIN refs_table ON lab_results.refs_id=refs_table.id  WHERE request_id=:a AND sex=:sex");
$result->bindParam(':a',$request_id);
$result->bindParam(':sex',$sex);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){

$parameter_name= $row['parameter_name'];
$normal_range= $row['normal_range'];
$results= $row['results'];


?>
<tr>
<input type="hidden" name="ref_id[]" value="<?php echo $ref_id; ?>">
<td><?php echo $parameter_name; ?></td>
<td><?php echo $normal_range; ?></td>
<td><?php echo $results; ?></td>	
<?php } ?>
</tr>
</tbody>
</table>
<?php } ?>
<?php

if (!isset($_GET["view"])){	
$patient=$_GET["patient"];
$test_id=$_GET["test_id"];
include 'sex.php';
$result = $db->prepare("SELECT*  FROM refs_table WHERE test_id=:a AND sex=:sex");
$result->bindParam(':a',$test_id);
$result->bindParam(':sex',$sex);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){

$parameter_name= $row['parameter_name'];
$normal_range= $row['normal_range'];
$ref_id= $row['id'];


?>
<tr>
<input type="hidden" name="ref_id[]" value="<?php echo $ref_id; ?>">
<td><?php echo $parameter_name; ?></td>
<td><?php echo $normal_range; ?></td>
<td><input type="text" name="result[]"></td>
<input type="hidden" name="patient" value="<?php echo $patient; ?>">
<input type="hidden" name="test_id" value="<?php echo $test_id; ?>">
<input type="hidden" name="request_id" value="<?php echo $_GET['request_id']; ?>">		
<?php } ?>
<input type="submit" value="submit" class="btn-success" style="width: 90%;" name="submit" />
</form>
<?php } ?>


