<?php 
include('../connect.php');
require_once('../main/auth.php');
if (isset($_GET['search'])) {
    $search=$_GET['search'];
    // code...
}
?>
<!DOCTYPE html>
<html>
<title>interim bill</title>
<?php 
include "../header.php"; 
?>
</head>
<body>
<style>
        @media screen {
   .onlyPrint {
       display: none;
   }
}
</style>
<header class="header clearfix" style="background-color: #3786d6;">

<?php include('../main/nav.php'); 
?>   
</header>
<?php include('side.php'); ?>
<div class="jumbotron" style="background: #95CAFC;">
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">patient bill</li>
<?php
include ('../connect.php');
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$a=$row['name'];
$b=$row['age'];
$c=$row['sex'];

?>
<?php if(isset($_GET['search']))
{ ?>
<li class="breadcrumb-item active" aria-current="page"><?php  echo $a; ?></li>
<li class="breadcrumb-item active" aria-current="page"> <?php 
include '../doctors/age.php';
?></li>
<?php } ?>
<?php } ?>
</ol>
</nav>
<form action="interim.php?" method="GET">
<span><?php
include("../pharmacy/patient_search.php");
?>
<input type="hidden" name="response" value="0"> <button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span> 
</form>  

<?php

if(isset($_GET['search']))
{
    ?>
<div>
<?php
$result = $db->prepare("SELECT * FROM settings");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$hospital=$row['name'];
$address=$row['address'];
$phone=$row['phone'];
$email=$row['email'];
$slogan=$row['slogan']; ?>
<div id="content" class="container"> 
<div  class="onlyPrint" align="center">  
<div class="logo-container" style="width: 20.3em; height: 10.4em;">
<img src="../icons/logo.jpg"  style="width: 100%; height: 100%;">
</div><!--End Info-->
<h6 ><?php echo $hospital; ?></h6>
<h6 ><?php echo $address; ?></h6>
<h6 ><?php echo $phone; ?></h6>
<h6 ><?php echo $email; ?></h6>
</hr>
<h6 ><?php echo $a; ?></h6>
<u><h5 >interim bill</h5></u>
<?php } ?>
</div>    

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

?>

<?php 
$patient=$_GET['search'];
$result = $db->prepare("SELECT drugs.drug_id,generic_name,brand_name,price*drugs.mark_up AS price,dispense_id,dispensed_drugs.quantity, dispensed_drugs.date AS date FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE patient='$patient' AND paid=0");
$result->execute();
$med_count = $result->rowcount();  
//Check whether the query was successful or not
if($med_count>0) {
?>
<label>medicines to be paid for</label></br> 
<table class="table" >
<thead class="bg-primary">
<tr>
    <th>Date</th>
<th>generic name</th>
<th>brand name</th>
<th>quantity</th>
<th>price</th>
<th>total</th>
</tr>
</thead>
<?php
if ($insurance==0) {
$result = $db->prepare("SELECT drugs.drug_id,token,generic_name,brand_name,price*drugs.mark_up AS price,dispense_id,dispensed_drugs.quantity,dispensed_drugs.date AS date FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE patient='$patient' AND paid=0");
}
if ($insurance==1) {
$result = $db->prepare("SELECT drugs.drug_id,token,generic_name,brand_name,price*drugs.ins_mark_up AS price,dispense_id,dispensed_drugs.quantity,dispensed_drugs.date AS date FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE patient='$patient' AND paid=0");
}
$result->execute();
for($i=0; $row = $result->fetch(); $i++){     
$drug = $row['generic_name'];
$brand = $row['brand_name'];
$price= round($row['price'],1);
$qty= $row['quantity'];
$drug_id= $row['drug_id'];
$token= $row['token'];
?>
<tbody> 
<tr>
    <td><?php echo date("d-m-Y", strtotime($row['date'])) ; ?></td>
<td><?php echo $drug; ?></td>
<td><?php echo $brand; ?></td>
<td><?php echo $qty; ?></td>
<td><?php echo ($price); ?></td>
<td ><?php  echo round(($qty*$price),1); ?></td>
<?php } ?>
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
<td> Total Amount: </td>      
</tr>
<tr>
<th colspan="4"><strong style="font-size: 12px; color: #222222;">Total:</strong></th>
<td colspan="1"><strong style="font-size: 12px; color: #222222;"> <?php $amount=$row['total'];  echo round($amount); ?></strong> </td>
</tbody>
</table><?php } ?>
<?php } ?>
<?php if($med_count<1) { 
$amount=0;
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
    <th>Date</th>
<th>test</th>
<th>requested by</th>
<th>cost</th>
</tr>
</thead>
<?php
if ($insurance==0) {
$result = $db->prepare("SELECT name,updated_at, test,opn,reqby,lab_tests.cost AS cost,lab.created_at AS date FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND  paid=0 ");
}
if ($insurance==1) {
$result = $db->prepare("SELECT name,updated_at, test,opn,reqby,lab_tests.ins_cost AS cost,lab.created_at AS date FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND  paid=0 ");
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
<td><?php echo date("d-m-Y", strtotime($row['date'])) ; ?></td>
<td><?php echo $name; ?></td>
<td><?php echo $reqby; ?></td>
<td><?php echo $cost; ?></td>
<?php }?>
</tr>
<tr> 
<?php 
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
<td> Total Amount: </td>
</tr>
<tr>
<th colspan="2"><strong style="font-size: 12px; color: #222222;">Total:</strong></th>
<td colspan="1"><strong style="font-size: 12px; color: #222222;"> <?php $amount_lab=$row['total']; echo $amount_lab; ?> </td><?php } ?>
</tbody>
</table>
</br> 
<?php } } ?>
<?php
if($lab_count<1) { 
$amount_lab=0;
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
<div class="container" > <label>clinics to be paid for</label></br> 
<table class="table" >
<thead class="bg-primary">
<tr>
<th>Date</th>
<th>clinic</th>
<th>cost</th>
</tr>
</thead>
<?php
$patient=$_GET['search'];
$result = $db->prepare("SELECT clinic_name, cost,clinic_fees.date AS date FROM clinics RIGHT OUTER JOIN clinic_fees ON clinic_fees.clinic_id=clinics.clinic_id WHERE patient='$patient' AND paid=0");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){     
$name = $row['clinic_name'];
$cost= $row['cost'];
?>
<tbody>
<tr>
<td><?php echo date("d-m-Y", strtotime($row['date'])) ; ?></td>
<td><?php echo $name; ?></td>
<td><?php echo $cost; ?></td>
<?php }?>
</tr>
<tr> 
<?php $patient=$_GET['search'];
$result = $db->prepare("SELECT sum(clinics.cost) as total FROM clinics RIGHT OUTER JOIN clinic_fees ON clinic_fees.clinic_id=clinics.clinic_id WHERE patient='$patient' AND paid=0");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){ ?>
<th>Total Amount: </th>
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
$amount_clinic=0;
?>
<?php } ?>
<?php
$b=$_GET['search'];
$d2=0;
$c=0;
$result = $db->prepare("SELECT * FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE paid_by=:a AND paid=:b AND collection.amount>:c");
$result->bindParam(':a', $b);
$result->bindParam(':b', $d2);
$result->bindParam(':c', $c);
$result->execute();
$fees_count = $result->rowcount();
if ($fees_count<1) {
$total_fees=0;
?>
<?php
}
if ($fees_count>=1) { 
?>
<h4>procedure charges</h4>
<table class="table" >
<thead class="bg-primary">
<tr>
    <th>Date</th>
<th>payment</th>
<th>amount</th>
</tr>
</thead>    
<?php
$b=$_GET['search'];
$d2=0;
$result = $db->prepare("SELECT fees_name,collection.amount AS amount,collection.date AS date FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE paid_by=:a AND paid=:b AND collection.amount>:c");
$result->bindParam(':a', $b);
$result->bindParam(':b', $d2);
$result->bindParam(':c', $d2);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){

?>
<tbody>
<tr> 
<td><?php echo date("d-m-Y", strtotime($row['date'])) ; ?></td>
<td><?php echo $row['fees_name']; ?>:&nbsp;</td>
<td> &nbsp;<?php echo $row['amount']; ?>

</td>
<?php } }
 ?>
</tbody>
</table>
<hr>
<?php
$result = $db->prepare("SELECT sum(collection.amount) AS total FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE paid_by=:a AND paid=:b");
$result->bindParam(':a', $b);
$result->bindParam(':b', $d2);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
?>
<table class="table" >
<thead class="bg-primary">
<tr>
<th>total</th>
<th>&nbsp;</th>
<th style="float: right;"><?php $total_fees=$row['total']; echo $total_fees;   ?></th>
<?php } ?>
</tr>
</thead> 
</table>
<?php
$result = $db->prepare("SELECT adm_date FROM admissions WHERE admissions.ipno='$search' AND discharged=0");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
    $w=$row['adm_date'];
}
    if (isset($w)) {
       include  'wardcal.php';
    }
       else{
        $admission_total=0;
       }
    ?>
<table class="table" >
<thead class="bg-primary">
<tr>
<th>grand total:</th>
<th width="70%">&nbsp;</th>
<th>
<?php

$grand_total=round($amount+$amount_lab+$amount_clinic+$total_fees+$admission_total); 

echo  $grand_total;     ?></th><?php

?>

</tr>
</thead> 
</table>
</tbody>
</div>
<button class="btn btn-success btn-large" style="margin-left: 45%;" value="content" id="goback" onclick="javascript:printDiv('content')" >print bill</button></br><p>&nbsp;</p>
<?php } ?>


</div>
</div>
</div>
<script type="text/javascript">
function printDiv(content) {
//Get the HTML of div
var divElements = document.getElementById(content).innerHTML;
//Get the HTML of whole page
var oldPage = document.body.innerHTML;

//Reset the page's HTML with div's HTML only
document.body.innerHTML = 
"<html><head><title></title></head><body>" + 
divElements + "</body>";

//Print Page
window.print();

//Restore orignal HTML
document.body.innerHTML = oldPage;          
}


</script>
<script src="dist/vertical-responsive-menu.min.js"></script>
</body>
</html>