<?php 
require_once('../main/auth.php');
include ('../connect.php');
?>
<!DOCTYPE html>
<html>
<title>payslip</title><meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
</head>
<body>
    <style>
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
        @media print {#amount {
  text-align: right;
}
#description {
  text-align: left;
}
}
    </style>
<header class="header clearfix" style="background-color: #3786d6;">
<?php 
include('../main/nav.php');
include "../header.php";
?>   
</header><?php include('sidee.php'); ?>
<div class="jumbotron" style="background: #95CAFC;">
<div class="container" id="content">
<div  class="onlyPrint" align="center"> 
<div class="container" align="center">
    <?php echo include_once "../header.php";?>
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
</div>
<?php 
$employee_id=$_GET['employee_id'];
$result = $db->prepare("SELECT*  FROM employees  JOIN job_groups ON employees.jg_id=job_groups.jg_id WHERE employee_id=:a");
$result->bindParam(':a',$employee_id);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){ 
$employee_id = $row['employee_id'];    
$name = $row['employee_name'];
$deployed_date=$row['date_deployed'];
$jg = $row['jg_name'];
$basic_pay = $row['basic_salary'];
$status = $row['status'];
?>
<ol class="breadcrumb">
<li class="breadcrumb-item active" aria-current="page">payslip</li>
<li class="breadcrumb-item active" aria-current="page"><?php echo $name;  ?></li>
<li class="breadcrumb-item active" aria-current="page">job group:</b> <?php echo $jg;  ?></li>
</ol><?php } ?>
<table class="table" >
<thead>
<tr>
<th></th>
<th></th>
</tr>
</thead>
<?php  
$period = date('Y-m');
$result = $db->prepare("SELECT*  FROM salaries_payments WHERE employee_id=:a AND CONCAT(YEAR(date),'-',RIGHT(CONCAT('0',MONTH(DATE)),2))=:b");
$result->bindParam(':a',$employee_id);
$result->bindParam(':b',$period);
$result->execute();
$row = $result->fetch(); 
$net_pay = $row['amount'];
$gross_pay = $row['gross_pay'];
$nfdw = $row['dw']/30;

?>
<tbody>
<tr>
<td style="width: 81.5%;">Basic pay</td>
<td id="amount"><?php echo $basic_pay; ?></td>
</tr>
</tbody>
</table>

<table class="table" >
<thead>
<tr>
<th>Allowances</th>
<th></th>
</tr>
</thead>
<?php  
$result = $db->prepare("SELECT *  FROM other_allowances  WHERE employee_id=:a AND CONCAT(YEAR(date),'-',RIGHT(CONCAT('0',MONTH(DATE)),2))=:b");
$result->bindParam(':a',$employee_id);
$result->bindParam(':b',$period);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){ 
$amount = $row['amount'];
$name = $row['name'];		
?>
<tr>
<td style="width: 81.5%;"><?php echo $name; ?></td>
<td><?php echo $amount; ?></td>
</tr>

<?PHP
}
?>
<tbody>
</tbody>
</table>


<table class="table" >
<thead>
<tr>
<th></th>
<th></th>
</tr>
</thead>
<tr> 
<tr>
<td style="width: 81.5%;">Gross pay</td>
<td><?php echo $gross_pay; ?></td>
</tr>
<tr> 
</tbody>
</table>


<table class="table">
<thead>
<tr>
<th>Deductions</th>
<th></th>
</tr>
</thead>
<?php   
$result = $db->prepare("SELECT* FROM nhif_payments WHERE employee_id=$employee_id AND CONCAT(YEAR(date),'-',RIGHT(CONCAT('0',MONTH(DATE)),2))=:a");
$result->bindParam(':a',$period);
$result->execute();
$nhif =$result->fetch()['amount'];
$nssf = 200;
?>
<tbody>
<tr>
<td style="width: 81.5%;">NHIF</td>
<td><?php echo $nhif; ?></td>
</tr>
<tr>
<td style="width: 81.5%;">NSSF</td>
<td><?php echo $nssf; ?></td>
</tr>
<tr> 
<?PHP 

$result = $db->prepare("SELECT*  FROM other_deductions WHERE employee_id=:a AND CONCAT(YEAR(date),'-',RIGHT(CONCAT('0',MONTH(DATE)),2))=:b");
$result->bindParam(':a',$employee_id);
$result->bindParam(':b',$period);
$result->execute();

for($i=0; $row = $result->fetch(); $i++){ 
$amount = $row['amount'];
$name = $row['name'];
?>
<tr>
<td><?PHP echo $name?></td>
<td><?PHP echo $amount?></td>
<tr> 
<?PHP 
}
?>
</tbody>
</table>
<table class="table">
<thead>
<tr>
<th></th>
<th></th>
</tr>
</thead>
<?php   
$result = $db->prepare("SELECT* FROM tax_paid WHERE employee_id=:a AND CONCAT(YEAR(date),'-',RIGHT(CONCAT('0',MONTH(DATE)),2))=:b");
$result->bindParam(':a',$_GET['employee_id']);
$result->bindParam(':b',$period);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){ 
$paye = $row['amount'];

?>
<tbody>
<tr>
<td style="width: 81.5%;">PAYE AUTO</td>
<td><?php  echo $paye;  ?></td>
<?php }?>

</tr>
</tbody>
</table>
<table class="table">
<thead>
<tr>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td style="width: 81.5%;"><b>Net pay</b></td>
<td><b><?php echo($net_pay); ?></b></td>
</tr>
<tr> 
</tbody>
</table>

</div>
<div><button value="content" id="goback" onclick="javascript:printDiv('content')">Print Payslip</button></div>
<script type="text/javascript">
function printDiv(content) {
//Get the HTML of div
var divElements = document.getElementById(content).innerHTML;
//Get the HTML of whole page
var oldPage = document.body.innerHTML;

//Reset the page's HTML with div's HTML only

document.body.innerHTML = 
"<html><head><title></title></head><body>"+
divElements + "</body>";

//Print Page
window.print();
//Restore orignal HTML
document.body.innerHTML = oldPage;          
}


</script>
</div>
</body>
</html>