<?php 
require_once('../main/auth.php');
$search=$_GET['search'];
$response=0;
include ('../connect.php');
?> 
<!DOCTYPE html>
<html>
<title>lab results</title>
<?php
include "../header.php";
?>
</head>

<body>

<header class="header clearfix" style="background-color: #3786d6;">
<?php include('../main/nav.php'); ?>

</header><?php include('side.php'); ?>
<div class="jumbotron" style="background: #95CAFC;">
<body onLoad="document.getElementById('country').focus();">
<nav aria-label="breadcrumb" style="width: 90%;">
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">lab patients</li>
<li class="breadcrumb-item active" aria-current="page">search patient</li>
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
<li class="breadcrumb-item active" aria-current="page"><?php echo $a; ?>&nbsp;  <?php include 'age.php'; ?>, &nbsp;  <?php echo $c; ?></li><?php } ?>
</ol>
</nav>
<form action="lab.php?" method="GET">
<span><?php
include "../pharmacy/patient_search.php";
?>
<input type="hidden" name="response" value="0"> <button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>     
</form> 
 
<?php
//checking if there were any lab tests requested for and if the patient has been served at lab 
$patient=$_GET['search'];
$served=1;
$result = $db->prepare("SELECT opn FROM lab  WHERE served=:a  AND opn=:b");
$result->BindParam(':a',$served);
$result->BindParam(':b',$patient);
$result->execute();
$rowcount = $result->rowcount();

if($rowcount>0) {
?>
<div class="container" > <label>lab tests requested for</label></br> 
<table class="resultstable" >
<thead>
<tr>
<th> request date</th>
<th>test</th>
<th>requested by</th>
<th>comments</th>
<th>view dails</th>

</tr>
</thead>
<?php
//if true for lab request, get the results

$result = $db->prepare("SELECT  lab.id AS id,template,name, test,opn,reqby,comments,created_at FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE (lab.served=:a)  AND opn=:b");
$result->BindParam(':a',$served);
$result->BindParam(':b',$patient);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$date = $row['created_at'];
$name = $row['name'];
$lab_id = $row['id'];
$reqby = $row['reqby'];
$comments= $row['comments'];
$template= $row['template'];

?>
<tbody>
<tr>
<td><?php echo $date; ?></td>
<td><?php echo $name; ?></td>
<td><?php echo $reqby; ?></td>
<td><?php echo $comments; ?></td>
<td><?php
// check if a template for the lab results is not empty
if (empty($template)){ ?>no details<?php } ?> 
<?php if (!empty($template)) {
# code...
?><a rel="facebox" href="lab_result.php?request_id=<?php echo $lab_id; ?>&patient=<?php echo $search; ?>&name=<?php echo $a; ?>&name=<?php echo $name; ?>&test_done=<?php echo $name; ?>&view=true">view details</a><?php } ?>
</td>

<?php }?>
</tr>

</tbody>
</table>
<?php }?>
<?php
//checking if there were any lab tests requested for and if the patient has been served at imaging 
$result = $db->prepare("SELECT opn FROM  req_images WHERE opn='$patient'");
$result->execute();
$rowcount = $result->rowcount();

if($rowcount>0) {
?>
</br>
<table class="table">
<tr>
<th>imaginging</th>
<th>requested by</th>
<th>request date</th>
<th> done on</th>
</tr>
<tr>
<?php 
$result = $db->prepare("SELECT* FROM req_images  RIGHT OUTER JOIN imaging ON req_images.test=imaging.imaging_id WHERE  opn='$patient' ");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){

$name = $row['imaging_name'];
$req_by = $row['reqby'];
$req_date = $row['created_at'];
$date_done= $row['updated_at'];
?>
<td><?php echo $name; ?></td>
<td><?php echo $req_by; ?></td>
<td><?php echo $req_date; ?></td>
<td><?php if ($date_done=="0000-00-00 00:00:00") {
echo "not yet done";
# code...
}
else{
echo $date_done;
} ?></td> <?php } ?>
</tr>
</table>
<?php
$result = $db->prepare("SELECT patient FROM images  WHERE  patient='$patient' LIMIT 1");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){     
$patient = $row['patient'];
if (!isset($patient)) {
echo "results for imaging yet to be posted";
}
if (isset($patient)) {

?>

<div class="container" >
<h4>image files</h4>
<table class="resultstable" >
<thead>
<tr>
<th>file</th>
<th>posted by</th>
<th>date</th>
<th>report</th>
<th>posted by</th>
<th>posted on</th>
<th>zip and download</th>
</tr>
</thead>
<?php
$result = $db->prepare("SELECT* FROM images  WHERE  patient='$patient'");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){     
$path = $row['image_path'];
$posted = $row['posted_by'];
$date = $row['date'];
$type = $row['type'];
$report = $row['report'];
$posted_by = $row['reported_by'];
$posted_date = $row['report_date'];
?>
<tbody>
<tr><td><?php if ($type==1) {

?><a href="../lab/imageloader/tests/index.php?id=<?php echo $path; ?>&pn=<?php echo $search; ?>&reload=1"><?php echo 'dicom_data'.$path.'- is dicom'; ?></a><?php } ?>
<?php if ($type==2) {  
?><a href="view_image.php?id=<?php echo '../lab/'.'imageloader/'.'tests/'.'data/'.$path; ?>&pn=<?php echo $search; ?>&reload=1"><?php echo 'image'.$path.'- is image'; ?></a><?php } ?></td>
<td><?php echo $posted; ?></td>
<td><?php echo $date; ?></td>
<td><?php if (!empty($report)) {
?><a rel="facebox" href="report.php?search=<?php echo $_GET['search']; ?>&path=<?php echo $path; ?>">     
<button class="btn btn-primary">view report</button><?php } ?></a>
<?php
if (empty($report)) {
echo "No Report Yet";

?><?php } ?>
</td>
<td><?php echo $posted_by; ?></td>
<td><?php echo $posted_date; ?></td>
<td><?php
$filename = $path.'.zip'; 

if (file_exists($filename)) {

?>
<a href="<?php echo $filename; ?>">
<button class="btn-success">download</button></a>
<?php } ?>
<?php 
if (!file_exists($filename)) {    
?>
<a href="zip.php?zip=<?php echo $path; ?>&name=<?php echo $search; ?>"><button class="btn-primary">make zip</button></a>

<?php } ?></td><?php } ?>
</tr>

</tbody>
</table>
<?php } ?><?php } } ?>
<hr>
<?php
$result = $db->prepare("SELECT * FROM prescriptions WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$cc=$row['cc'];
$hpi=$row['hpi'];
if (!empty($cc || $hpi)) {
# code...

?>
<center>diagnosis and prescriptions</center>
<div class="container" id="content">
<table class="resultstable">
<thead>
<tr>
<th>cc</th>
<th>hpi</th>
<th>date</th>
</tr>
</thead>
<?php
$result = $db->prepare("SELECT * FROM prescriptions WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$cc=$row['cc'];
$hpi=$row['hpi'];
$datep=$row['date'];     
?>
<tbody>
<tr>  
<td><?php echo $cc;?></td>
<td><?php if (empty($hpi)) {
echo "no history taken";    
}
else{
echo $hpi ;
} ?></td>
<td><?php echo $datep;?></td>
<?php }?>
</tr>
</tbody>
</table>
<?php }?>
<?php }?>
<?php
$result = $db->prepare("SELECT * FROM ddx WHERE patient=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$code=$row['disease'];
if (!empty($code)) {

?>
<hr>
<label>differential diagnosis</label>
<table class="resultstable"  >
<thead>
<tr>
<th>date</th>
<th>differential diagnosis</th>
</tr>
</thead>
<?php
$result = $db->prepare("SELECT * FROM ddx JOIN icd_second_level_codes ON ddx.disease=icd_second_level_codes.code  WHERE patient=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$ddx=$row['title'];
$datep=$row['date'];     
?>
<tbody>
<tr>
<td><?php echo $datep;?></td>
<td><?php echo $ddx;?></td>
<?php }?>
</tr>
</tbody>
</table>
<?php }?>
<?php }?>
<p></p>
<?php
$result = $db->prepare("SELECT * FROM lab WHERE opn=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$patient=$row['opn'];
if (!empty($patient)) {

?>
<center>previous lab requests</center>
<table class="table" >
<thead>
<tr>
<th> request date</th>
<th>test</th>
<th>requested by</th>
<th>comments</th>
<th>view dails</th>
<th>tick as seen</th>
</tr>
</thead>
<?php
//if true for lab request, get the results
$patient=$_GET['search'];
$result = $db->prepare("SELECT  lab.id AS id,template,name, test,opn,reqby,comments,created_at,lab.served AS served FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' ");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$date = $row['created_at'];
$name = $row['name'];
$lab_id = $row['id'];
$reqby = $row['reqby'];
$comments= $row['comments'];
$template= $row['template'];
$served= $row['served'];
?>
<tbody>
<tr>
<td><?php echo $date; ?></td>
<td><?php echo $name; ?></td>
<td><?php echo $reqby; ?></td>
<td><?php echo $comments; ?></td>
<td><?php
if ($template == 0) {
?>no details template<?php
}
?>
<?php
if (($template == 1)) {
# code...
?><a rel="facebox" href="lab_result.php?request_id=<?php  echo $lab_id; ?>&view=true&name=<?php echo $name; ?>">view details</a><?php
}
?></td>
<td><?php
if ($served==2) {
    // code...
?>
<form id="form" action="change.php" method="POST">
    <input type="checkbox" value="<?php  echo $lab_id; ?>" onchange="$('#form').submit();" name="id" class="checkbox"/>
</form>
<?php } ?></td>
<?php }?>
</tr>

</tbody>
</table>
<?php }?>

<p></p>
<?php
$result = $db->prepare("SELECT* FROM patient_notes  WHERE patient=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$created_at=$row['created_at'];
if (!empty($patient)) {

?>
<center>patient notes</center>
<table class="resultstable"  >
<thead>
<tr>
<th>date</th>
<th>details</th>
</tr>
</thead>
</thead>
<?php $result = $db->prepare("SELECT* FROM patient_notes  WHERE patient=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$date=$row['created_at'];
$notes=$row['notes'];
?>
<tbody>
<tr>

<td><?php echo $date; ?></td>
<td><?php echo $notes;  ?><?php } ?></td>
</tr>
</tbody>
</table> 
</tr>
</tbody>
</table>
<?php }?>
<?php }?>
<?php
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$name=$row['name'];
if (!empty($name)) {
# code...
?>
<a  href="admitted.php?search=<?php echo $patient; ?>&islab=1"><button class="btn btn-success">plan or further investigation</button></a>
<p>&nbsp;</p>
<a href="prescribe_inp.php?search=<?=$search; ?>&code=<?=rand(); ?>&response=0"><button class="btn btn-success"> prescribe drugs</button></a>
<?php } ?>
<?php }?>
</div></div></div><?php } ?></div></div></div></div></div>

</body>
</html>