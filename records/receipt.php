<?php 
include('../connect.php');
require_once('../main/auth.php');
$_GET['search']=$_GET['number'];
?> 
<!DOCTYPE html>
<html>
<title>registration receipt</title>
<?php 
include "../header.php"; 
?>
</head>
<style>
#amount {
  text-align: right;
}
#description {
  text-align: left;
}
    #invoice-POS{
  box-shadow: 0 0 1in -0.25in rgba(0, 0, 0, 0.5);
  padding:2mm;
  margin: 0 auto;
  width: 44mm;
  background: #FFF;
  
  
}
 
 #mid,#bot{ /* Targets all id with 'col-' */
  border-bottom: 1px solid #EEE;
}


#mid{min-height: 80px;} 
#bot{ min-height: 50px;}


.info{
  display: block;
  //float:left;
  margin-left: 0;
}
.title{
  float: right;
}
.title p{text-align: right;} 
table{
  width: 100%;
  border-collapse: collapse;
}
td{
  //padding: 5px 0 5px 15px;
  //border: 1px solid #EEE
}
.tabletitle{
  //padding: 5px;
  font-size: .5em;
  background: #EEE;
}

</style>
<style>
.letter {
display:none;
}
</style>
<script>
 function printDiv() {
  // Get the HTML content of the div
  var divContents = document.getElementById("content").innerHTML;

  // Create a new window and set its content to the div
  var printWindow = window.open('', '', '');
  printWindow.document.write('<html><head><title>Print Div Example</title>');
  printWindow.document.write('</head><body>');
 printWindow.document.write(divContents);
  printWindow.document.write('</body></html>');
  printWindow.document.close();

  // Print the window
  printWindow.print();
}
</script>

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
<li class="breadcrumb-item active" aria-current="page">patient bill</li>
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

<div class="container" id="content">
<div class="letter">
<div   align="center">  
<div class="logo-container" style="width: 20.3em; height: 10.4em;">
<img src="../icons/logo.jpg"  style="width: 100%; height: 100%;"></div>
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
<h6><?php echo $_GET['name'];  ?></h6>
<p><?php echo date("D, d/m/Y"); ?></p>
</div>
</div>

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
$insurance=$row['type'];

$patient=$_GET['search'];
$result = $db->prepare("SELECT drugs.drug_id,generic_name,brand_name,price*drugs.mark_up AS price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE patient='$patient' AND paid=0");
$result->execute();
$med_count = $result->rowcount();  
//Check whether the query was successful or not
if($med_count>0) {
?>
<label>medicines to be paid for</label></br> 
<table class="table" >
<thead class="bg-primary">
<tr>
<th id="description">generic name</th>
<th id="description">brand name</th>
<th id="description">quantity</th>
<th id="description">price</th>
<th id="amount" style="text-align: right;" >total</th>
</tr>
</thead>
<?php
if ($insurance==0) {
$result = $db->prepare("SELECT drugs.drug_id,token,generic_name,brand_name,price*drugs.mark_up AS price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE patient='$patient' AND paid=0");
}
if ($insurance==1) {
$result = $db->prepare("SELECT drugs.drug_id,token,generic_name,brand_name,price*drugs.ins_mark_up AS price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE patient='$patient' AND paid=0");
}
$result->execute();
for($i=0; $row = $result->fetch(); $i++){     
$drug = $row['generic_name'];
$brand = $row['brand_name'];
$price= $row['price'];
$qty= $row['quantity'];
$drug_id= $row['drug_id'];
$token= $row['token'];
?>
<tbody> 
<tr>
<td id="description"><?php echo $drug; ?></td>
<td id="description"><?php echo $brand; ?></td>
<td id="description"><?php echo $qty; ?></td>
<td id="description"><?php echo ($price); ?></td>
<td id="amount" style="text-align: right;"><?php  echo ($qty*$price); ?></td><?php } ?>
</tr>
<tr> 
<?php 
if ($insurance==0) {
$result = $db->prepare("SELECT sum(price*dispensed_drugs.quantity*drugs.mark_up) as total FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id =dispensed_drugs.drug_id WHERE patient='$patient' AND paid=0");
}
if ($insurance==1) {
$result = $db->prepare("SELECT sum(price*dispensed_drugs.quantity*drugs.ins_mark_up) as total FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id =dispensed_drugs.drug_id WHERE patient='$patient' AND paid=0");
}
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
?>
<th> </th>
<th>  </th>
<th>  </th>
<th>  </th>
<td id="description" style="text-align: left;"> Total Amount: </td>      
</tr>
<tr>
<th colspan="3"><strong style="font-size: 12px; color: #222222;">Total:</strong></th>
<td colspan="1"><strong style="font-size: 12px; color: #222222;"> <?php $amount=$row['total'];  echo round($amount); ?></strong> </td>
</tbody>
</table><?php } ?><?php } ?>
</br>
<?php if($med_count<1) { 
?>
<?php } ?>
<?php 

$result = $db->prepare("SELECT name, test,opn,reqby,lab_tests.cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND  paid=0");
$result->execute();
$lab_count = $result->rowcount();

//Check whether the query was successful or not
if($lab_count>0) {
?>
<label>lab tests to be paid for</label></br> 
<table class="table" >
<thead class="bg-primary">
<tr>
<th id="description">test</th>
<th id="description">requested by</th>
<th id="amount">cost</th>
</tr>
</thead>
<?php
if ($insurance==0) {
$result = $db->prepare("SELECT name,updated_at, test,opn,reqby,lab_tests.cost AS cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND  paid=0 ");
}
if ($insurance==1) {
$result = $db->prepare("SELECT name,updated_at, test,opn,reqby,lab_tests.ins_cost AS cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND  paid=0 ");
}
$result->execute();
for($i=0; $row = $result->fetch(); $i++){

$name = $row['name'];
$reqby = $row['reqby'];
$cost= round($row['cost']);
$updated= $row['updated_at'];

?>
<tbody>
<tr>
<td id="description"><?php echo $name; ?></td>
<td id="description"><?php echo $reqby; ?></td>
<td id="amount"><?php echo $cost; ?></td>
<?php }?>
</tr>
<tr> <?php 
if ($insurance==0) {
$result = $db->prepare("SELECT sum(lab_tests.cost) as total FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND paid=0");
}
if ($insurance==1) {
$result = $db->prepare("SELECT sum(lab_tests.ins_cost) as total FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND paid=0");
}
$result->execute();
for($i=0; $row = $result->fetch(); $i++){ ?>
<th> </th>
<th>  </th>
<td id="description"> Total Amount: </td>
</tr>
<tr>
<th id="description"><strong style="font-size: 12px; color: #222222;">Total:</strong></th>
<td id="amount"><strong style="font-size: 12px; color: #222222;"> <?php $amount_lab=$row['total']; echo $amount_lab; ?> </td><?php } ?>
</tbody>
</table>
</br> 
<?php }  ?>
<?php
if($lab_count<1) { 
?>
<?php } ?> 
<?php 
$patient=$_GET['search'];
$result = $db->prepare("SELECT clinic_name, cost FROM clinics RIGHT OUTER JOIN clinic_fees ON clinic_fees.clinic_id=clinics.clinic_id WHERE patient='$patient' AND paid=0");
$result->execute();
$clinic_count = $result->rowcount();

//Check whether the query was successful or not
if($clinic_count>0) {
?>
<label>clinics to be paid for</label></br> 
<table class="table" >
<thead class="bg-primary">
<tr>
<th id="description">clinic</th>
<th id="amount">cost</th>
</tr>
</thead>
<?php
$patient=$_GET['search'];
$result = $db->prepare("SELECT clinic_name, cost FROM clinics RIGHT OUTER JOIN clinic_fees ON clinic_fees.clinic_id=clinics.clinic_id WHERE patient='$patient' AND paid=0");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){     
$name = $row['clinic_name'];
$cost= $row['cost'];
?>
<tbody>
<tr>
<td id="description" style="text-align:left;"><?php echo $name; ?></td>
<td id="amount" style="text-align: right;"><?php echo $cost; ?></td>
<?php }?>
</tr>
<tr> <?php $patient=$_GET['search'];
$result = $db->prepare("SELECT sum(clinics.cost) as total FROM clinics RIGHT OUTER JOIN clinic_fees ON clinic_fees.clinic_id=clinics.clinic_id WHERE patient='$patient' AND paid=0");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){ ?>
<td>Total Amount: </td>
<td> <?php $amount_clinic=$row['total'];
echo $amount_clinic; ?> </td>
</tr>
<?php } ?>
</tbody>
</table>
</br> 
<?php }  ?>
<?php
if($clinic_count<1) { 
?>
<?php } ?>
<?php
$b=$_GET['search'];
$d2=0;
$result = $db->prepare("SELECT* FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE paid_by=:a AND paid=:b");
$result->bindParam(':a', $b);
$result->bindParam(':b', $d2);
$result->execute();
$fees_count = $result->rowcount();
if ($fees_count<0) {
# code...

?>
<?php
}
if ($fees_count>0) { 
?>
<table class="table" >
<thead class="bg-primary">
<tr>
<th id="description" style="text-align:left;">description</th>
<th id="amount" style="text-align: right;">amount</th>
</tr>
</thead>    
<?php
$b=$_GET['search'];
$d2=0;
$result = $db->prepare("SELECT fees_name,collection.amount AS amount FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE paid_by=:a AND paid=:b");
$result->bindParam(':a', $b);
$result->bindParam(':b', $d2);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){

?>
<tbody>
<tr> 
<td id="description" style="text-align:left;"><?php echo $row['fees_name']; ?>:&nbsp;</td>
<td id="amount" style="text-align: right;"> &nbsp;<?php echo $row['amount']; ?></td><?php } ?></tbody>
</table>
<hr>
<table class="table" >
<thead class="bg-primary">
<?php
$result = $db->prepare("SELECT sum(collection.amount) AS total FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE paid_by=:a AND paid=:b");
$result->bindParam(':a', $b);
$result->bindParam(':b', $d2);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
?>

<th id="description">total</th>
<th id="amount" style="text-align: right;"><?php $total_fees=$row['total']; echo $total_fees;   ?></th>
</tr>
</table>
<?php } ?>
<?php } ?>
<?php
$reset=1;
$result = $db->prepare("SELECT charges AS charges,adm_date,discharge_date FROM admissions RIGHT OUTER JOIN wards ON wards.id=admissions.ward WHERE ipno=:a AND discharged=:b");
$result->bindParam(':a', $search);
$result->bindParam(':b', $reset);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$discharge_date=$row['discharge_date'];
if ($discharge_date="0000-00-00") {
$discharge_date=date("Y-m-d");
}
$adm_date=$row['adm_date'];
$startdate = strtotime($adm_date);
$enddate = strtotime($discharge_date);
$datediff = $enddate - $startdate;
$days=round($datediff / (60 * 60 * 24));
if (isset($days)) {
?>
</tr><?php 

?>
<table class="table" >
<thead class="bg-primary">
<tr>
<th id="description">total for admission</th>
<th>&nbsp;</th>
<th id="amount"><?php $admission_total=$days*$row['charges']; echo $admission_total;   ?></th>
</tr>
</thead> 
</table>
<?php } ?>
<?php } ?>
<table class="table" >
<thead class="bg-primary">
<tr>
<th id="description">grand total:</th>
<th width="70%">&nbsp;</th>
<th><?php if (!isset($amount)) 
{ $amount=0;   # code...
} 
if (!isset($amount_lab)) {
$amount_lab=0;   # code...
}
if (!isset($amount_clinic)) {
$amount_clinic=0;
# code...
}
if (!isset($total_fees)) { 
$total_fees=0; }

if (!isset($admission_total)) { 
$admission_total=0; 
}
if (!isset($token)) { 
$token=""; 
}

?>
<?php
$grand_total=round($amount+$amount_lab+$amount_clinic+$total_fees+$admission_total); }
if (isset($grand_total)) {
# code...
echo  $grand_total; }    ?></th>
</tr>
</thead> 
</table>
</tbody>


</div></div></div>

<script src="dist/vertical-responsive-menu.min.js"></script>

</body>
</html>