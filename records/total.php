<?php 
require_once('../main/auth.php');
?>

<!DOCTYPE html>
<html>
<title>visits</title><?php 
include('../header.php');
?>

</head><body onLoad="document.getElementById('country').focus();">
<header class="header clearfix" style="background-color: #3786d6;;">


</button>
<?php include('../main/nav.php'); ?>

</header><?php include('side.php'); ?>
<div class="content-wrapper">   
<div class="jumbotron" style="background: #95CAFC;">         
<link rel="stylesheet" href="../main/jquery-ui.css">
<script src="../main/jquery-1.12.4.js"></script>
<script src="../main/jquery-ui.js"></script>
<script>
$( function() {
$( "#date_one" ).datepicker({
changeMonth: true,
changeYear: true
});
} );

</script>
<script>
$( function() {
$( "#date_two" ).datepicker({
changeMonth: true,
changeYear: true
});
} );

</script>

</head>
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">patient visit report</li>
<li class="breadcrumb-item active" aria-current="page" style="float: right;"><a href="visits.php"> visits for period</a></li>
<li class="breadcrumb-item active" aria-current="page" style="float: right;"><a href="patients.php"> patients for period</a></li>
</ol>
<span>
<div class="container">
<div class="container" id="content">
<div class="container">
<p>&nbsp;</p>
<center>showing patients</center>
<p>&nbsp;</p>
</div>
<table class="table" >
<thead>
<tr>  <th>date</th>
<th>patient name</th>
<th>number</th>
<th>age</th>
</tr>
</thead>
<tbody>
<?php
include('../connect.php');
$result = $db->prepare("SELECT name,opno,age FROM  patients");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){       

?>
<tr> <td><?php echo date("d-M, Y", strtotime($row['date'])); ?>&nbsp;</td>
<td><?php echo $row['name']; ?>&nbsp;</td>
<td> &nbsp;<?php echo $row['opno']; ?>
<td>&nbsp;<?php
         $now = time('Y/m/d');
$dob = strtotime($row['age']);
$datediff = $now - $dob;
$agee=round($datediff / (60 * 60 * 24))/365; 
$age = number_format($agee, 1, '.', '');

if ($age>=1) {
  echo $age." years";
   # code...
 }
 if ($age<1) {
  echo $age*12; echo "&nbsp;"."Months";
    # code...
  } ?></td>
</td><?php }  ?></tbody>
</table>
<table class="table" >
<thead>
<tr>  <th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
<th>&nbsp;</th>
</tr>
</thead>
<tbody>
<?php     
$result = $db->prepare("SELECT count(opno) AS total FROM patients");

$result->execute();
for($i=0; $row = $result->fetch(); $i++){       

?>
<tr> <td>total</td>
<td>&nbsp;</td>
<td> &nbsp;</td>
<td> &nbsp;<?php echo $row['total']; ?>
</td><?php }  ?></tbody>
</table>
</div>
</div>
<button class="btn btn-success btn-large" style="margin-left: 45%;" value="content" id="goback" onclick="javascript:printDiv('content')" >print report</button>

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

<script src="../pharmacy/dist/vertical-responsive-menu.min.js"></script>
</body>
</html>