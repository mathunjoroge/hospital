<?php 
include('../connect.php');
require_once('../main/auth.php');
$d1=date('Y-m-d')." 00:00:00";
$d2=date('Y-m-d H:i:s');

$receipt=$_GET['receipt'];

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
<title>cashier</title>
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
  
  
::selection {background: #f31544; color: #FFF;}
::moz-selection {background: #f31544; color: #FFF;}
h1{
  font-size: 1.5em;
  color: #222;
}
h2{font-size: .9em;}
h3{
  font-size: 1.2em;
  font-weight: 300;
  line-height: 2em;
}
p{
  font-size: .7em;
  color: #666;
  line-height: 1.2em;
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
<?php 
include "../header.php"; 
?>
</head>
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
$receipt=$_GET['receipt'];
$result = $db->prepare("SELECT * FROM settings");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$hospital=$row['name'];
$address=$row['address'];
$phone=$row['phone'];
$email=$row['email'];
$slogan=$row['slogan']; ?>
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
<button class="btn btn-success btn-large" style="margin-left: 45%;" value="content" id="goback" onclick="javascript:printDiv('content')" >print receipt</button></br><p>&nbsp;</p>
<div id="content" class="container"> 
 <div  class="container" align="center">  
 <div class="logo-container" style="width: 20.3em; height: 10.4em;">
<img src="../icons/logo.jpg"  style="width: 100%; height: 100%;">
</div><!--End Info-->
</center><!--End InvoiceTop-->
<div id="mid">
<div class="info">
<h6 ><?php echo $hospital; ?></h6>
<h6 ><?php echo $address; ?></h6>
<h6 ><?php echo $phone; ?></h6>
<h6 ><?php echo $email; ?></h6>
</hr>

<h6 ></h6>
<h6 >payment mode: <?php echo $mode; ?></b></h6>
<?php } ?>
</div>
</div>
</div><!--End Invoice Mid-->
<div id="bot">
<table class="table table-bordered" >
<thead class="bg-primary">
<tr>
<th>generic name</th>
<th>brand name</th>
<th>quantity</th>
<th>price</th>
<th>total</th>
<th>action</th</tr>
</thead>
<?php
$token=$_GET['receipt'];
$insurance=0;
if ($insurance==0) {
$result = $db->prepare("SELECT drugs.drug_id AS drug,generic_name,brand_name,price*drugs.mark_up AS price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE token=:b");
}
if ($insurance==1) {
$result = $db->prepare("SELECT drugs.drug_id AS drug,generic_name,brand_name,price*drugs.ins_mark_up AS price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE token=:b");
}
$result->BindParam(':b', $token);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){

$drug = $row['generic_name'];
$brand = $row['brand_name'];
$price= $row['price'];
$qty= $row['quantity'];
$drug_id= $row['drug'];
?>
<tbody>
<tr>
<td><?php echo $drug; ?></td>
<td><?php echo $brand; ?></td>
<td ><?php echo $qty; ?></td>
<td><?php echo round($price); ?></td>
<td ><?php  echo round($qty*$price); ?></td>
<?php }?>
</tr>
<tr> 
<?php
if ($insurance==0) {
$result = $db->prepare("SELECT sum(price*dispensed_drugs.quantity*drugs.mark_up) as total FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id =dispensed_drugs.drug_id WHERE  token=:b");
}
if ($insurance==1) {
$result = $db->prepare("SELECT sum(price*dispensed_drugs.quantity*drugs.ins_mark_up) as total FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id =dispensed_drugs.drug_id WHERE  token=:b");
}
$result->BindParam(':b', $token);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){ ?>

</tr>
<tr>
<th colspan="4"><strong style="font-size: 12px; color: #222222;">Total:</strong></th>
<td colspan="1" id="myvalue"><strong style="font-size: 12px; color: #222222;"> <?php $total=round($row['total']); echo $total; ?> </td><?php } ?>
</tbody>
</table>
</div>
</div><!--End Table-->
</div><!--End InvoiceBot-->
</div><!--End Invoice-->      
</div>
</div>

<script src="dist/vertical-responsive-menu.min.js"></script>

</body>
</html>