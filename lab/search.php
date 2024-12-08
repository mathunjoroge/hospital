<?php 
require_once('../main/auth.php');
include('../connect.php');
$result = $db->prepare("SELECT * FROM lab_orders");
$result->execute();
$rowcountt = $result->rowcount();
$rowcount = $rowcountt+1;
$code=$rowcount;
?>
<!DOCTYPE html>
<html>
<title>search patient</title>
<?php
include "../header.php";
?>
</head>
<style type="text/css">
table.blueTable {
border: 1px solid #1C6EA4;
background-color: #EEEEEE;
width: 70%;
text-align: left;
border-collapse: collapse;
}
table.blueTable td, table.blueTable th {
border: 1px solid #AAAAAA;
padding: 3px 2px;
}
table.blueTable tbody td {
font-size: 13px;
}
table.blueTable tr:nth-child(even) {
background: #D0E4F5;
}
table.blueTable thead {
background: #1C6EA4;
background: -moz-linear-gradient(top, #5592bb 0%, #327cad 66%, #1C6EA4 105%);
background: -webkit-linear-gradient(top, #5592bb 0%, #327cad 66%, #1C6EA4 105%);
background: linear-gradient(to bottom, #5592bb 0%, #327cad 66%, #1C6EA4 105%);
border-bottom: 2px solid #444444;
}
table.blueTable thead th {
font-size: 15px;
font-weight: bold;
color: #FFFFFF;
border-left: 2px solid #D0E4F5;
}
table.blueTable thead th:first-child {
border-left: none;
}

table.blueTable tfoot td {
font-size: 14px;
}
table.blueTable tfoot .links {
text-align: right;
}
table.blueTable tfoot .links a{
display: inline-block;
background: #1C6EA4;
color: #FFFFFF;
padding: 2px 8px;
border-radius: 5px;
}.column {
float: left;
width: 50%;
}

/* Clear floats after the columns */
.row:after {
content: "";
display: table;
clear: both;
}
</style>
<header class="header clearfix" style="background-color: #3786d6;">
</button>
<?php include('../main/nav.php'); ?>

</header><?php include('side.php'); ?>
<div class="jumbotron" style="background: #95CAFC;">
<nav aria-label="breadcrumb" style="width: 90%;">
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">lab patients</li>
<li class="breadcrumb-item active" aria-current="page">waiting list</li>
<li class="breadcrumb-item"><a href="edit.php?search= &response=0">edit results</a></li>
<?php
if (isset($_GET['search'])) {
$search = $_GET['search'];
include('../connect.php');
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for ($i = 0; $row = $result->fetch(); $i++) {
$a = $row['name'];
$b = $row['age'];
$c = $row['sex'];
$d = $row['opno'];
?>
<li class="breadcrumb-item active" aria-current="page"><?php
echo $a;
?> <?php include "../doctors/age.php"; ?> &nbsp; </caption></li><?php
}
?>sex: <?php echo $c; ?> <?php } ?>
</ol>
</nav>

<p>&nbsp;</p>
<caption align="left">  </caption> 
<form action="search.php?" method="GET">
<span><?php
include "../pharmacy/patient_search.php";
?>
<input type="hidden" name="response" value="0"> <button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span> 
</form>
<?php
if (isset($_GET['search'])) { 
?>
<p>&nbsp;</p>
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
$served=1;
$patient=$_GET['search'];
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
?><a rel="facebox" href="lab_result.php?request_id=<?php echo $lab_id; ?>&patient=<?php echo $search; ?>&name=<?php echo $name; ?>&view=true">view details</a><?php } ?>
</td>
<?php }?>
</tr>

</tbody>
</table>
</br>

<input type="hidden" name="id" value="<?php echo $_GET['search']; ?>"> 
<button class="btn btn-success btn-large" style="width: 100%;">save</button></a> <?php }  ?></form></div>



<script src="../pharmacy/dist/vertical-responsive-menu.min.js"></script>
</div></div></div></div>

</body>
</html>