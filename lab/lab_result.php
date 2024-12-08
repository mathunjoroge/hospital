<?php 
include('../connect.php');

$search = $_GET['patient'];
include('../connect.php');
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for ($i = 0; $row = $result->fetch(); $i++) {
$a = $row['name'];
$b = $row['age'];
$c = $row['sex'];
$d = $row['opno'];
}
?>
<script>
function printDiv() {
// Get the HTML content of the div
var divContents = document.getElementById("content").innerHTML;

// Create a new window and set its content to the div
var printWindow = window.open('', '', 'width=800');
printWindow.document.write('<html><head><title>Print Div Example</title>');
printWindow.document.write('</head><body>');
printWindow.document.write(divContents);
printWindow.document.write('</body></html>');
printWindow.document.close();

// Print the window
printWindow.print();
}
</script>
<style>
.letter {
display:none;
}
</style>
<div class="container" id="content" style="width: 40.3em;" >
<div class="letter"
<div   align="center">  
<div class="logo-container" style="width: 20.3em; height: 10.4em;">
<img src="../icons/logo.jpg"  style="width: 100%; height: 100%;">
</div>
<?php
$result = $db->prepare("SELECT * FROM settings");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$hospital=$row['name'];
$address=$row['address'];
$phone=$row['phone'];
$email=$row['email'];
$slogan=$row['slogan']; ?>
<h6 ><?php echo $hospital; ?></h6>
<h6 ><?php echo $address; ?></h6>
<h6 ><?php echo $phone; ?></h6>
<h6 ><?php echo $email; ?></h6>
<?php } ?>
</div>
<div   align="center">
<h1><?php echo $_GET['name'];  ?></h1>
<p>Report generated on <?php echo date("D, d/m/Y"); ?></p>

<p>Patient Name: <?php echo $a;  ?></p>
<p>Lab Technician: <?php echo $_SESSION['SESS_LAST_NAME']; ?></p>
</div>
<div   align="center">
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
$result = $db->prepare("SELECT*  FROM lab_results RIGHT OUTER JOIN refs_table ON lab_results.refs_id=refs_table.id  WHERE request_id=:a ");
$result->bindParam(':a',$request_id);
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
</div>
<?php } ?>

</div></div>

<button onclick="printDiv()">Print</button>

