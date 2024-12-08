<?php 
require_once('../main/auth.php');
include('../connect.php');
$token=$_GET['receipt'];
$_GET['search']='%20';
?>
<!DOCTYPE html>
<html>
<title>pharmacy</title>
<?php
  include "../header.php";
  ?>
</head>
<body>
<header class="header clearfix" style="background-color: #3786d6;">
<?php include('../main/nav.php'); ?>   
</header>
<?php include('side.php'); ?>
<div class="content-wrapper" style=" background-image: url('../images/doctor.jpg');">
<div class="jumbotron" style="background: #95CAFC;">
</header>
<div class="content-wrapper">   
<div>
<nav aria-label="breadcrumb" style="width: 90%;">
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">patient</li>
<li class="breadcrumb-item active" aria-current="page">search patient</li>
<li style="float:right;"><a href="waiting.php?list=3&token=<?php echo mt_rand(10000000, 99999999); ?>&search=%20&response=0">oncology waiting list</a></li>
<li style="float:right;"><a href="waiting.php?list=2&token=<?php echo mt_rand(10000000, 99999999); ?>&search=%20&response=0">inpatient waiting list</a></li>
<li style="float:right;"><a href="waiting.php?list=1&token=<?php echo mt_rand(10000000, 99999999); ?>&search=%20&response=0">outpatient waiting list</a></li></ol>


</nav>

<label>select medicines for patient</label></br>
<table class="table table-bordered" style="width:80%;">
<thead>
<tr>
<th>drug name</th>
<th>price</th>
<th >qty availabe</th>
<th>qty </th>
<th>add</th>
</tr>
</thead>
</table>   
<span><form action="savewalkin.php" method="POST">
<input type="hidden" name="token" value="<?php echo $token; ?>">
<tbody>
<tr>
<td>
<select id="medicine" name="med"  data-live-search="true" class="selectpicker" data-live-search="true" title="Please select a medicine..." onchange="showDrug(this.value)" style="width:20rem;" required>
<?php 

$result = $db->prepare("SELECT * FROM drugs");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
   echo "<option value=".$row['drug_id'].">".$row['generic_name']."-".$row['brand_name']."</option>";
 }

?>      
</select>
</td>
<b><span id="med"></b><button class="btn btn-success btn-large">add</button></form></span></div>      
<div class="container" id="results" > <label>selected meds</label></br> 
<?php
$patient=$_GET['search'];
$result = $db->prepare("SELECT drugs.drug_id,generic_name,brand_name,price,dispense_id,dispensed_drugs.quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE token=:b");
$result->BindParam(':b', $token);
$result->execute();
$rowcount = $result->rowcount();

if ($rowcount>0) {
# code...

?>
<table class="table table-bordered" >
<thead class="bg-primary">
<tr>
<th>generic name</th>
<th>brand name</th>
<th>quantity</th>
<th>price</th>
<th>total</th>
<th>action</th>
</tr>
</thead>
<?php
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
<td><a rel="facebox" href="editqtyw.php?id=<?php echo $row['dispense_id']; ?>&qty=<?php echo $qty; ?>&gname=<?php echo $drug; ?>&bname=<?php echo $brand; ?>&admitted=<?php echo $admitted; ?>&pn=<?php echo $pn; ?>&did=<?php echo $drug_id; ?>&token=<?php echo $_GET["receipt"]; ?>&rs=1"><button class="btn btn-success" style="height: 5px;" title="Click to edit quantity"></button></a><a href="deletew.php?id=<?php echo $row['dispense_id']; ?>&pn=<?php echo $pn; ?>&admitted=<?php echo $admitted; ?>&did=<?php echo $drug_id; ?>&qty=<?php echo $qty; ?>&token=<?php echo $_GET["receipt"]; ?>"> <button class="btn btn-danger" style="height: 5px;" title="Click to Delete"></button></a> </td><?php }?>
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
</br>
<a href="save.php?receipt=<?php echo $token; ?>&total=<?php echo $total; ?>">
<button class="btn btn-success btn-large" style="width: 100%;">save</button></a></div>
</div>      
</div>
<?php } ?>
</div> 
<script>
function showDrug(str) {
  if (str == "") {
    document.getElementById("med").innerHTML = "";
    return;
  } else {
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
        document.getElementById("med").innerHTML = this.responseText;
      }
    };
    xmlhttp.open("GET","get_drug.php?q="+str,true);
    xmlhttp.send();
  }
}
</script>
</body>
</html>