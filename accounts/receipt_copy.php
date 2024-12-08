<?php 
include('../connect.php');
require_once('../main/auth.php');
include('../header.php');
 $patient=$_GET['search'];
 ?> 
 <!DOCTYPE html>
<html>
<title>cashier</title>
 
</head>
<body>
    <style>#amount {
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
 
#top, #mid,#bot{ /* Targets all id with 'col-' */
  border-bottom: 1px solid #EEE;
}

#top{min-height: 100px;}
#mid{min-height: 80px;} 
#bot{ min-height: 50px;}

#top .logo{
  //float: left;
  height: 60px;
  width: 60px;
  background-size: 60px 60px;
}
.clientlogo{
  float: left;
  height: 60px;
  width: 60px;
  background: url(http://michaeltruong.ca/images/client.jpg) no-repeat;
  background-size: 60px 60px;
  border-radius: 50px;
}
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
.service{border-bottom: 1px solid #EEE;}
.item{width: 24mm;}
.itemtext{font-size: .5em;}

#legalcopy{
  margin-top: 5mm;
}

  
  
}
</style>
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
      $search=$_GET['search'];
      include ('../connect.php');
      $result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
       $result->BindParam(':o', $search);
        $result->execute();
        for($i=0; $row = $result->fetch(); $i++){
        $a=$row['name'];
     
     ?>
     <li class="breadcrumb-item active" aria-current="page"><?php echo $a; ?></li><?php } ?>
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
      <button class="btn btn-success btn-large" style="margin-left: 45%;" value="content" id="goback" onclick="javascript:printDiv('content')" >print receipt</button>
<div class="jumbotron" >
<div id="content"> 
<div class="container" align="center">
<div id="invoice-POS">    
<center id="top">
<div class="logo"></div>
<div class="info">  
  
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
  <div class="logo-container" style="width: 20.3em; height: 10.4em;">
<img src="../icons/logo.jpg"  style="width: 100%; height: 100%;">
</div>
</div><!--End Info-->
</center><!--End InvoiceTop-->
</div>
<div id="mid">
<div class="info">
    <h6 ><?php echo $hospital; ?></h6>
    <h6 ><?php echo $address; ?></h6>
    <h6 ><?php echo $phone; ?></h6>
    <h6 ><?php echo $email; ?></h6>
    </hr>
     <h6 ><?php echo $a; ?></h6>
     <h6 >P/N: <?php echo $search; ?></h6>
<?php }
echo "invoice number: ". $receipt.'</br>'; ?>
<label>mode of payment: <?php $mode=$_GET['mode'];
if ($mode==1) {
  echo "cash"."</br>";
  # code...
}
if ($mode==2) {
  echo "mobile money"."</br>";
  # code...
}
if ($mode==3) {
  echo "insurance"."</br>";
  }

if ($mode==4) {
  echo "bank"."</br>";
  # code...
}
 ?></label></br>
<?php

 if ($_GET['insurance']>0) {
    $insurance=$_GET['insurance'];
    $result = $db->prepare("SELECT * FROM insurance_companies WHERE company_id=:a");
    $result->BindParam(':a', $insurance);
        $result->execute();
        for($i=0; $row = $result->fetch(); $i++){
        $company=$row['name'];
        $ins_mark_up=$row['ins_mark_up'];
        
    echo "<label>"."invoice to: ".$company."</lable>";
  }
}

  $ins_mark_up=1;

 ?>
 </div>
</div><!--End Invoice Mid-->

<div id="bot">

  <?php 
$patient=$_GET['search'];
$result = $db->prepare("SELECT drugs.drug_id,generic_name,brand_name,price*drugs.mark_up*dispensed_drugs.mark_up AS price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE receipt_no=:a");
        $result->bindParam(':a',$receipt);
        $result->execute();
        $med_count = $result->rowcount();  
  //Check whether the query was successful or not
    if($med_count>0) {
?>
<table class="table">
<thead>
<tr>
<th class="item">generic name</th>
<th class="Hours">brand name</th>
<th class="rate">quantity</th>
<th class="rate">price</th>
<th id="amount">total</th>
</tr>
</thead>
<?php
      if ($insurance==0) {
        $result = $db->prepare("SELECT drugs.drug_id,token,generic_name,brand_name,price*drugs.mark_up  AS price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE receipt_no=:a");
        }
        if ($insurance==1) {
        $result = $db->prepare("SELECT drugs.drug_id,token,generic_name,brand_name,price*drugs.ins_mark_up  AS price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE receipt_no=:a");
        }
        $result->bindParam(':a',$receipt);
  $result->execute();
  for($i=0; $row = $result->fetch(); $i++){     
      $drug = $row['generic_name'];
      $brand = $row['brand_name'];
      $price= round($row['price']);
      $qty= $row['quantity'];
      $drug_id= $row['drug_id'];
      $token= $row['token'];
         ?>
<tbody> 
<tr>
<td id="description"><?php echo $drug; ?></td>
<td><?php echo $brand; ?></td>
<td><?php echo $qty; ?></td>
<td><?php echo $price; ?></td>
<td id="amount"><?php  echo $qty*$price; ?></td><?php } ?>
</tr>
<tr> <?php
if ($insurance==0) {
        $result = $db->prepare("SELECT sum(price*dispensed_drugs.quantity*drugs.mark_up) as total FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id =dispensed_drugs.drug_id WHERE receipt_no=:a");
        }
        if ($insurance==1) {
        $result = $db->prepare("SELECT sum(price*dispensed_drugs.quantity*drugs.ins_mark_up) as total FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id =dispensed_drugs.drug_id WHERE receipt_no=:a");
        }
        $result->bindParam(':a',$receipt);
  $result->execute();
  for($i=0; $row = $result->fetch(); $i++){
   ?>
      <th> </th>
      <th>  </th>
      <th>  </th>
      <th>  </th>
      <td id="description"> Total Amount: </td>      
    </tr>
      <tr>
        <th colspan="4"><strong style="font-size: 12px; color: #222222;" id="description">Total:</strong></th>
        <td colspan="1" id="amount"><strong style="font-size: 12px; color: #222222;"> <?php $amount=round($row['total']);  echo ($amount); ?></strong> </td>
</tbody>
</table><?php } ?><?php } ?>
 </br>
 <?php if($med_count<1) { 
  ?>

<?php } ?>
<?php
$result = $db->prepare("SELECT name, test,opn,reqby,lab_tests.cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE  receipt_no=:a");
        $result->bindParam(':a',$receipt);
        $result->execute();
        $lab_count = $result->rowcount();
  
  //Check whether the query was successful or not
    if($lab_count>0) {
?>
<div class="container" > <label>lab tests  paid for</label></br> 
     <table class="table" >
<thead>
<tr>
<th id="description">test</th>
<th>requested by</th>
<th id="amount">cost</th>
</tr>
</thead>
<?php
      if ($insurance==0) {
        $result = $db->prepare("SELECT name, test,opn,reqby,lab_tests.cost AS cost FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE  receipt_no=:a");
        }
        if ($insurance==1) {
        $result = $db->prepare("SELECT name, test,opn,reqby,lab_tests.ins_cost AS cost  FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE  receipt_no=:a");
        }
        $result->bindParam(':a',$receipt);
        $result->execute();
  for($i=0; $row = $result->fetch(); $i++){
     
      $name = $row['name'];
      $reqby = $row['reqby'];
      $cost= $row['cost']*$ins_mark_up;
     

         ?>
<tbody>
<tr>
<td id="description"><?php echo $name; ?></td>
<td><?php echo $reqby; ?></td>
<td id="amount"><?php echo $cost; ?></td>
<?php }?>
</tr>
<tr> <?php
if ($insurance==0) {
        $result = $db->prepare("SELECT sum(lab_tests.cost) as total FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE receipt_no=:a");
        }
        if ($insurance==0) {
        $result = $db->prepare("SELECT sum(lab_tests.ins_cost) as total FROM lab RIGHT OUTER JOIN lab_tests ON lab.test=lab_tests.id WHERE receipt_no=:a");
        }
        $result->bindParam(':a',$receipt);
  $result->execute();
  for($i=0; $row = $result->fetch(); $i++){ ?>
      <th> </th>
      <th>  </th>
      <td id="description"> Total Amount: </td>
      </tr>
      <tr>
        <th colspan="2"><strong style="font-size: 12px; color: #222222;">Total:</strong></th>
        <td colspan="1" id="amount"><strong style="font-size: 12px; color: #222222;"> <?php $amount_lab=$row['total']*$ins_mark_up; echo $amount_lab; ?> </td><?php } ?>
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
$result = $db->prepare("SELECT clinic_name, cost FROM clinics RIGHT OUTER JOIN clinic_fees ON clinic_fees.clinic_id=clinics.clinic_id WHERE receipt_no=:a");
        $result->bindParam(':a',$receipt);
        $clinic_count = $result->rowcount();
  
  //Check whether the query was successful or not
    if($clinic_count>0) {
?>
<div class="container" > <label>clinics to be paid for</label></br> 
     <table class="table" >
<thead>
<tr>
<th id="description">clinic</th>
<th id="amount">cost</th>
</tr>
</thead>
<?php
      
        $result = $db->prepare("SELECT clinic_name, cost FROM clinics RIGHT OUTER JOIN clinic_fees ON clinic_fees.clinic_id=clinics.clinic_id WHERE receipt_no=:a");
        $result->bindParam(':a',$receipt);
  $result->execute();
  for($i=0; $row = $result->fetch(); $i++){     
      $name = $row['clinic_name'];
      $cost= $row['cost'];
         ?>
<tbody>
<tr>
<td id="description"><?php echo $name; ?></td>
<td id="amount"><?php echo $cost; ?></td>
<?php }?>
</tr>
<tr> <?php
        $result = $db->prepare("SELECT sum(clinics.cost) as total FROM clinics RIGHT OUTER JOIN clinic_fees ON clinic_fees.clinic_id=clinics.clinic_id WHERE receipt_no=:a");
        $result->bindParam(':a',$receipt);
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
        $result = $db->prepare("SELECT fees_name,collection.amount AS amount FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE receipt_no=:a");
        $result->bindParam(':a',$receipt);
         $result->execute();
        $fees_count = $result->rowcount();
        if ($fees_count<0) {
          # code...
                 
 ?>
 <p> no fees to be paid for</p>
 <?php
 }
        if ($fees_count>0) { 
  ?>
<table class="table" >
<thead>
<tr>
    <th id="description">description</th>
      <th id="amount">amount</th>
    </tr>
</thead>    
      <?php
      $b=$_GET['search'];
        $d2=0;
        $result = $db->prepare("SELECT fees_name,fees.amount AS alt_amount,collection.amount AS amount FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id  WHERE receipt_no=:a");
        $result->bindParam(':a',$receipt);
        $result->execute();
        for($i=0; $row = $result->fetch(); $i++){
                
      ?>
      <tbody>
<tr> <td id="description"><?php echo $row['fees_name']; ?>:&nbsp;</td>
      <td id="amount"> &nbsp;<?php if (isset($row['amount'])){
      echo $row['amount']*$ins_mark_up;
          
      } 
      else{
          echo $row['alt_amount']*$ins_mark_up;
      }?>

</td><?php } ?></tbody>
</table>
<hr>
<?php
 $result = $db->prepare("SELECT sum(collection.amount) AS total FROM collection RIGHT OUTER JOIN fees ON fees.fees_id=collection.fees_id WHERE receipt_no=:a");
        $result->bindParam(':a',$receipt);
        $result->execute();
        for($i=0; $row = $result->fetch(); $i++){
 ?>
<table class="table" >
<thead class="bg-primary">
<tr>
<th>total</th>
<th>&nbsp;</th>
<th id="amount"><?php $total_fees=$row['total']; echo $total_fees;   ?></th>
</tr>
</thead> 
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
<thead>
<tr>
    <th id="description">total for admission</th>
    <th>&nbsp;</th>
      <th id="amount"><?php $admission_total=$days*$row['charges']*$ins_mark_up; echo $admission_total;   ?></th>
    </tr>
</thead> 
 </table>
 <?php } ?>
<?php } ?>
<table class="table" >
<thead>
<tr>
    <th id="description">grand total:</th>
    <th width="70%">&nbsp;</th>
      <th id="amount"><?php if (!isset($amount)) 
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
 
 <table class="table" >
<thead>
<tr>
    <th id="description">&nbsp;</th>
      <th id="amount">&nbsp;</th>
    </tr>
</thead>
<tr>
 <td id="description">amount tendered:</td>

<td id="amount">
  
<?php

  ?></td>
</tr>

 </table>
 <?php if (isset($grand_total)) {
   # code...
  ?>
 
 <?php } ?>
 <?php if ($_GET['mode']==3) {
   # code...
 ?>
 <label>patient name:  <?php echo $a; ?>: signature:............................................................... </label>
<?php } ?>
</div><!--End Table-->

</div><!--End InvoiceBot-->
</div><!--End Invoice-->
 

<script src="dist/vertical-responsive-menu.min.js"></script>

</body>
</html>