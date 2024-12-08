<?php
include('../connect.php');
if (isset($_GET['test_name'])) {
?>
<h4 align="center"><?php echo $_GET['test_name']; ?>&nbsp;<?php
$sex=$_GET['sex'];

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
 ?></h4>
<form action="submit.php" method="POST">
<table class="resultstable" >
<thead>
<tr>
<th>parameter</th>
<th>normal range</th>
</tr>
</thead>
<tbody>
<?php
//lab id is unique id on lab table and will be used to update the template	
$lab_id=$_GET["test_id"];
$sex=$_GET["sex"];
$result = $db->prepare("SELECT*  FROM refs_table WHERE test_id=:a AND sex=:b ");

$result->bindParam(':a',$lab_id);
$result->bindParam(':b',$sex);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){

$parameter_name= $row['parameter_name'];
$normal_range= $row['normal_range'];

?>
<tr>
<input type="hidden" name="ref_id[]" value="<?php echo $ref_id; ?>">
<td><?php echo $parameter_name; ?></td>
<td><?php echo $normal_range; ?></td>

<?php } ?>
</tr>
</tbody>
</table>
<?php } ?>