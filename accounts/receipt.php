<?php 
require_once('../main/auth.php');
include('../connect.php');
$receipt_no=$_GET['receipt'];
?> 
<!DOCTYPE html>
<html>
<title>receipt</title>
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
@media print {
  .noPrint {
      display:none;
  }
}
@media screen {
   .onlyPrint {
       display: none;
   }
}
</style>
<style>
.letter {
display:none;
}
</style>
<header class="header clearfix" style="background-color: #3786d6;">
<?php include('../main/nav.php'); ?>   
</header>
<?php include('side.php'); ?>
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
<?php
require_once('../main/auth.php');
$mode=$_GET['mode'];
$d1=date('Y-m-d')." 00:00:00";
$d2=date('Y-m-d H:i:s');
$search=$_GET['search'];
$receipt=$_GET['receipt'];
$patient=$_GET['search'];
$payment_mode=$_GET['mode'];

if ($payment_mode==1) {
$mode='cash';
}
if ($payment_mode==2) {
$mode='mobile money';
}
if ($payment_mode==3) {
$mode='insurance';
}
if ($payment_mode==4) {
$mode='bank';
}
?> 
<!DOCTYPE html>
<html>
<title>receipt</title>
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
$receipt=$_GET['receipt'];
$result = $db->prepare("SELECT * FROM settings");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$hospital=$row['name'];
$address=$row['address'];
$phone=$row['phone'];
$email=$row['email'];
$slogan=$row['slogan']; ?>
<button class="btn btn-success btn-large" style="margin-left: 45%;" value="content" id="goback" onclick="javascript:printDiv('content')" >print receipt</button></br><p>&nbsp;</p>
<div id="content" class="container"> 
<div  class="onlyPrint" align="center">  
<h4 ><i><?php echo $hospital; ?></i></h4>
<p><i>Excellent Care Closer Home</i></p>
<h6 ><?php echo $address; ?></h6>
<h6 ><?php echo $phone; ?></h6>
<h6 ><?php echo $email; ?></h6>
</hr>
<h6 ><?php echo $a; ?></h6>
<h6 ><b>P/N: <?php echo $search; ?></h6>
<h6 >payment mode: <?php echo $mode; ?></b></h6>
<?php } ?>
</div>
<!--End Invoice Mid-->
<div id="bot">
<?php 
$patient=$_GET['search'];
$result = $db->prepare("SELECT drugs.drug_id,generic_name,brand_name,price*drugs.mark_up AS price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE patient='$patient' AND receipt_no=$receipt");
$result->execute();
$med_count = $result->rowcount();  
//Check whether the query was successful or not
if($med_count>0) {
?>
<label>medicines  paid for</label></br> 
<table class="table" >
<thead class="bg-primary">
<tr>
<th>description</th>
<th>quantity</th>
<th>price</th>
<th>total</th>
</tr>
</thead>
<?php
if ($insurance==0) {
$result = $db->prepare("SELECT drugs.drug_id,token,generic_name,brand_name,price*drugs.mark_up AS price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE patient='$patient' AND receipt_no=$receipt");
}
if ($insurance==1) {
$result = $db->prepare("SELECT drugs.drug_id,token,generic_name,brand_name,price*drugs.ins_mark_up AS price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE patient='$patient' AND receipt_no=$receipt");
}
$result->execute();
for($i=0; $row = $result->fetch(); $i++){     
$drug = $row['generic_name'];
$brand = $row['brand_name'];
$price= round(($row['price']),1);
$qty= $row['quantity'];
$drug_id= $row['drug_id'];
$token= $row['token'];
?>
<tbody> 
<tr>
<td><?php if (isset($brand)) {
echo $brand;
// code...
}
else{
echo $drug;
} ?></td>
<td><?php echo $qty; ?></td>
<td><?php echo ($price); ?></td>
<td ><?php  echo ($qty*$price); ?></td><?php } ?>
</tr>
<tr> 
<?php 
if ($insurance==0) {
$result = $db->prepare("SELECT sum(price*dispensed_drugs.quantity*drugs.mark_up) as total FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id =dispensed_drugs.drug_id WHERE patient='$patient' AND receipt_no=$receipt");
}
if ($insurance==1) {
$result = $db->prepare("SELECT sum(price*dispensed_drugs.quantity*drugs.ins_mark_up) as total FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id =dispensed_drugs.drug_id WHERE patient='$patient' AND receipt_no=$receipt");
}
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
?>
<th> </th>
<th>  </th>
<th>  </th>
</tr>
<tr>
<th colspan="3"><strong style="font-size: 12px; color: #222222;">Total:</strong></th>
<td colspan="1"><strong style="font-size: 12px; color: #222222;"> <?php $amount=round(($row['total']),1);  echo round($amount,1); ?></strong> </td>
</tbody>
</table><?php } ?><?php } ?>
</br>
<?php if($med_count<1) { 
?>

<?php } ?>
<?php 

$result = $db->prepare("SELECT name, test,opn,reqby,lab_tests.cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND  receipt_no=$receipt");
$result->execute();
$lab_count = $result->rowcount();

//Check whether the query was successful or not
if($lab_count>0) {
?>

<label>lab tests  paid for</label></br> 
<table class="table" >
<thead class="bg-primary">
<tr>
<th>test</th>
<th>cost</th>
</tr>
</thead>
<?php
if ($insurance==0) {
$result = $db->prepare("SELECT name,updated_at, test,opn,reqby,lab_tests.cost AS cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND  receipt_no=$receipt ");
}
if ($insurance==1) {
$result = $db->prepare("SELECT name,updated_at, test,opn,reqby,lab_tests.ins_cost AS cost  FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND  receipt_no=$receipt ");
}
$result->execute();
for($i=0; $row = $result->fetch(); $i++){

$name = $row['name'];
$reqby = $row['reqby'];
$cost= round($row['cost'],1);
$updated= $row['updated_at'];

?>
<tbody>
<tr>
<td><?php echo $name; ?></td>
<td><?php echo $cost; ?></td>
<?php }?>
</tr>
<tr> <?php 
if ($insurance==0) {
$result = $db->prepare("SELECT sum(lab_tests.cost) as total FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND receipt_no=$receipt");
}
if ($insurance==1) {
$result = $db->prepare("SELECT sum(lab_tests.ins_cost) as total FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE opn='$patient' AND receipt_no=$receipt ");
}
$result->execute();
for($i=0; $row = $result->fetch(); $i++){ ?>
<th> </th>
<th>  </th>

</tr>
<tr>
<th colspan="3"><strong style="font-size: 12px; color: #222222;">Total:</strong></th>
<th colspan="3"><strong style="font-size: 12px; color: #222222;"> <?php $amount_lab=$row['total']; echo $amount_lab; ?> </th><?php } ?>
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
$result = $db->prepare("SELECT clinic_name, cost FROM clinics RIGHT OUTER JOIN clinic_fees ON clinic_fees.clinic_id=clinics.clinic_id WHERE patient='$patient' AND receipt_no=$receipt");
$result->execute();
$clinic_count = $result->rowcount();

//Check whether the query was successful or not
if($clinic_count>0) {
?>
<label>clinics  paid for</label></br> 
<table class="table" >
<thead class="bg-primary">
<tr>
<th>clinic</th>
<th>cost</th>
</tr>
</thead>
<?php
$patient=$_GET['search'];
$result = $db->prepare("SELECT clinic_name, cost FROM clinics RIGHT OUTER JOIN clinic_fees ON clinic_fees.clinic_id=clinics.clinic_id WHERE patient='$patient' AND receipt_no=$receipt");
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
<tr> <?php $patient=$_GET['search'];
$result = $db->prepare("SELECT sum(clinics.cost) as total FROM clinics RIGHT OUTER JOIN clinic_fees ON clinic_fees.clinic_id=clinics.clinic_id WHERE patient='$patient' AND receipt_no=$receipt");
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
?>

<?php } ?>
<?php
$b=$_GET['search'];
$d2=1;
$result = $db->prepare("SELECT* FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE paid_by=:a AND receipt_no=:b");  
$result->bindParam(':a', $b);
$result->bindParam(':b', $receipt_no);
$result->execute();
$fees_count = $result->rowcount();
if ($fees_count<0) {
# code...

?>
<?php
}
if ($fees_count>0) { 
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
$c=0;
$result = $db->prepare("SELECT fees_name,collection.amount AS amount FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE paid_by=:a AND receipt_no=:b AND collection.amount>:c");
$result->bindParam(':a', $b);
$result->bindParam(':b', $receipt_no);
$result->bindParam(':c', $c);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){

?>
<tbody>
<tr> <td><?php echo $row['fees_name']; ?>:&nbsp;</td>
<td> &nbsp;<?php echo $row['amount']; ?>

</td><?php } ?></tbody>
</table>
<hr>
<?php
$result = $db->prepare("SELECT sum(collection.amount) AS total FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE paid_by=:a AND receipt_no=:b");
$result->bindParam(':a', $b);
$result->bindParam(':b', $receipt_no);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
?>
<table class="table" >
<thead class="bg-primary">
<tr>
<th>total</th>
<th>&nbsp;</th>
<th><?php $total_fees=$row['total']; echo $total_fees;   ?></th>
</tr>
</thead> 
</table>

<?php } ?>
<?php } ?>

</tr><?php 

?>
<?php
if ($_GET['du']>0) {
    
  ?>
  <table class="table" >
<thead class="bg-primary">
<tr>
<th>total for ward charges</th>
<th> <?php echo $_GET['du'].' days'; ?></th>
<th><?php $admission_total=$_GET['ward']; echo $admission_total;   ?></th>
</tr>
</thead> 
</table>

<?php }?>

<?php } ?>
<table class="table" >
<thead class="bg-primary">
<tr>
<th>grand total:</th>
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
$grand_total=($amount+$amount_lab+$amount_clinic+$total_fees+$admission_total); 

# code...
echo  $grand_total;  ?></th>
</tr>
</thead> 
</table>
</tbody><?php if (isset($grand_total)) {
# code...
?>
<?php } ?>
<p>you were served by: <?php echo $_SESSION['SESS_FIRST_NAME']; ?> </p>
<p>
<?php
if (isset($_GET['insurance'])) {
// code...

$company_id=$_GET['insurance'];
$result = $db->prepare("SELECT * FROM insurance_companies WHERE company_id =:company_id ");
$result->BindParam(':company_id', $company_id );
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$insurance=$row['name'];
echo 'invoice to: '.$insurance;
}
}

?>
<p><?php echo date('d M, Y'); ?></p>
</div>
    </div>
    </div>
<script src="dist/vertical-responsive-menu.min.js"></script>

</body>
</html>