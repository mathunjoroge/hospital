<?php 
require_once('../main/auth.php');
?>
<!DOCTYPE html>
<html>
<title>Procedure fees</title>
<head>
<?php
include "../header.php";
?>
</head>
<body>
<header class="header clearfix" style="background-color: #3786d6;">

</button>
<?php include('../main/nav.php'); ?>

</header><?php include('side.php'); ?>
  
<div class="jumbotron" style="width:auto;background: #95CAFC;">
     <?php
    if (isset($_GET['success'])) {

    ?>
    <script>
        setTimeout(function() {
    $('#alert').fadeOut('fast');
}, 3000); 
    </script>
    <div id="alert">
        <p class="text text-success">procedure has been saved</p>
    </div>
    <?php } ?>
<nav aria-label="breadcrumb" style="width: 90%;">
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">patient</li>
<li class="breadcrumb-item active" aria-current="page">procedure fees</li>
<li class="breadcrumb-item active" aria-current="page">
   
<?php
if (isset($_GET['search'])) {
# code...
$search=$_GET['search'];
include ('../connect.php');
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$a=$row['name'];
$b=$row['age'];
$c=$row['sex'];
$d=$row['opno'];

echo $a; ?>:  &nbsp;<?php  echo $c; ?>, 
<?php include 'age.php';  ?>
<?php } } ?>
</nav> 
<form action="clinical.php?&response=0" method="GET">
  <span><?php
  include "../pharmacy/patient_search.php";
  ?>
  <input type="hidden" name="token" value="<?php echo $_GET['token']; ?>">
  <input type="hidden" name="response" value="0"> <button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>     
</form>

   <?php
if (isset($_GET['search'])) {
?>
<div class="container-fluid">
 
<h3>add procedure charges</h3>
<form action="save_clinical.php" method="POST">        
<table class="table">
<tr>
<th>service</th>
<th>amount</th>
<th>add</th>
</tr>
<tr>
<?php
$result = $db->prepare("SELECT*  FROM  fees WHERE is_nursing=2");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$id = $row['fees_id'];
$fee = $row['fees_name']; 
$amount =$row['amount'];  

?>
<td><?php echo $fee; ?></td>
<td><?php echo $amount; ?></td>
<input type="hidden" name="patient" value="<?php echo $_GET['search']; ?>">
<input type="hidden" name="token" value="<?php echo $_GET['token']; ?>">
<td><input type="checkbox" name="fees[]" value="<?php echo $id; ?>"></td>
</tr>
<?php } ?>
</table>
<button class="btn btn-success btn-large" style="width: 70%;">add selected</button></form><?php }   ?>
</div>

<?php
if (isset($_GET['resp'])) {
    ?>
<div class="container-fluid">
<h3>save procedure charges</h3>
<form action="save_edited_clinical.php" method="POST"> 
<label>done by</label>
<select id="professional" name="done_by" class="form-select form-control" required/>
    <option >--select from the list--</option>
  
<?php 
        $result = $db->prepare("SELECT * FROM professionals");
        $result->execute();
        for($i=0; $row = $result->fetch(); $i++){                   
        ?> 
        <option value="<?php echo $row['profession_id']; ?>"><?php echo $row['name']; ?></option>
        <?php } ?>     
</select>
<table class="table">
<tr>
<th>service</th>
<th>amount(this part is editable)</th>
</tr>
<tr>
<?php
$token=$_GET['token'];
$result = $db->prepare("SELECT collection.amount AS amount, collection.collection_id AS id, fees_name  FROM  collection RIGHT OUTER JOIN fees ON collection.fees_id=fees.fees_id WHERE token=:token");
$result->BindParam(':token', $token);
$result->execute(); 

for($i=0; $row = $result->fetch(); $i++){
$fee=$row['fees_name'];
$amount =$row['amount']; 
$id =$row['id'];


?>
<td><?php echo $fee; ?></td>
<td><input value="<?php echo $amount; ?>" name="amount[]" contenteditable="true" type="number" required/></td>
<input type="hidden" name="collection_id[]" value="<?php echo $id; ?>">


</tr>
<?php } ?>
</table>
<button class="btn btn-success btn-large" style="width: 70%;">save</button></form><?php }   ?> 
</div>
</div>

<script src="../pharmacy/dist/vertical-responsive-menu.min.js"></script>

</body>
</html>