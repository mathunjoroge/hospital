<?php 
include('../connect.php');
require_once('../main/auth.php');
?>
<!DOCTYPE html>
<html>
<title>cashier</title>
<?php 

?>
</head>
<body>
<header class="header clearfix" style="background-color: #3786d6;">

<?php include('../main/nav.php');
include "../header.php"; 
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
$insurance=$row['type'];

?>

<?php 
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
<th>generic name</th>
<th>brand name</th>
<th>quantity</th>
<th>price</th>
<th>total</th>
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
$price= round($row['price'],1);
$qty= $row['quantity'];
$drug_id= $row['drug_id'];
$token= $row['token'];
?>
<tbody> 
<tr>
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
<h6 class="alert alert-warning" style="width: 43%;">no medicines prescribed yet or the patient has not gone to pharmacy</h6>
<?php } ?>
<?php 

$result = $db->prepare("SELECT name, test,opn,reqby,lab_tests.cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND  paid=0 AND  (lab.served=2 OR  lab.served=3)");
$result->execute();
$lab_count = $result->rowcount();

//Check whether the query was successful or not
if($lab_count>0) {
?>
<label>lab tests to be paid for</label></br> 
<table class="table" >
<thead class="bg-primary">
<tr>
<th>test</th>
<th>requested by</th>
<th>cost</th>
</tr>
</thead>
<?php
if ($insurance==0) {
$result = $db->prepare("SELECT name,updated_at, test,opn,reqby,lab_tests.cost AS cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND  paid=0 AND  (lab.served=2 OR  lab.served=3)");
}
if ($insurance==1) {
$result = $db->prepare("SELECT name,updated_at, test,opn,reqby,lab_tests.ins_cost AS cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND  paid=0 AND  (lab.served=2 OR  lab.served=3)");
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
<td><?php echo $name; ?></td>
<td><?php echo $reqby; ?></td>
<td><?php echo $cost; ?></td>
<?php }?>
</tr>
<tr> 
<?php 
if ($insurance==0) {
$result = $db->prepare("SELECT sum(lab_tests.cost) as total FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND paid=0 AND  (lab.served=2 OR  lab.served=3)");
}
if ($insurance==1) {
$result = $db->prepare("SELECT sum(lab_tests.ins_cost) as total FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND paid=0 AND  (lab.served=2 OR   lab.served=3)");
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
<h6 class="alert alert-warning" style="width: 43%;">no lab tests requested</h6>
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
<th>clinic</th>
<th>cost</th>
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
<h6 class="alert alert-warning" style="width: 43%;">no clinic requested yet</h6>
<?php } ?>
<?php
$b=$_GET['search'];
$d2=0;
$result = $db->prepare("SELECT * FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE paid_by=:a AND paid=:b");
$result->bindParam(':a', $b);
$result->bindParam(':b', $d2);
$result->execute();
$fees_count = $result->rowcount();
if ($fees_count<1) {
$total_fees=0;
?>
<h6 class="alert alert-warning" style="width: 43%;"> no fees to be paid for</h6>
<?php
}
if ($fees_count>=1) { 
?>
<h4>procedure charges</h4>
<table class="table" >
<thead class="bg-primary">
<tr>
<th>payment</th>
<th>amount</th>
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
<tr> <td><?php echo $row['fees_name']; ?>:&nbsp;</td>
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
<th><?php $total_fees=$row['total']; echo $total_fees;   ?></th>
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
        $days=0;
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
<a rel="facebox" href="pay.php?id=<?php echo $_GET['search']; ?>&med_amount=<?php echo $amount; ?>&lab=<?php echo $amount_lab; ?>&clinic=<?php echo $amount_clinic; ?>&fees=<?php echo $total_fees; ?>&total=<?php echo $grand_total; ?>&wards_income=<?php echo $admission_total; ?>&du=<?php echo $days; ?>">
<button class="btn btn-success btn-large" style="width: 100%;">save</button></a>

<script src="dist/vertical-responsive-menu.min.js"></script>
</body>
</html>