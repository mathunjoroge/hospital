<?php 
require_once('../main/auth.php');
include ('../connect.php');
?> 
<!DOCTYPE html>
<html>
<title>
<?php
$search=$_GET['search'];
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$a=$row['name'];
echo "discharge summary"." for ".$a;
}
if (($_GET["search"]==0)) {
echo "search patient";
}

?>
</title>
<?php
include "../header.php";
?>
</head>
<body>
<header class="header clearfix" style="background-color: #3786d6;;">


</button>
<?php include('../main/nav.php'); ?>

</header><?php include('side.php'); ?>
<div class="content-wrapper">   
<div class="jumbotron" style="background: #95CAFC;">
<nav aria-label="breadcrumb" style="width: 90%;">
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">out patient</li>
<li class="breadcrumb-item active" aria-current="page">discharge patient</li>
</nav>
<?php
$respose=$_GET['response'];

if ($respose==1) {

?>
<script>
setTimeout(function() {
$('#alert').fadeOut('fast');
}, 3000); 
</script>
<div id="alert" style="width: 40%;">
<p class="text text-success">discharge note saved</p>
</div><?php } ?>
<?php
if (!empty($_GET["search"])) {
$search=$_GET['search'];
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$a=$row['name'];
$b=$row['age'];
$c=$row['sex'];
$d=$row['opno'];
?>

<li class="breadcrumb-item active" aria-current="page"><?php echo $a; ?> &nbsp; age:  <?php include "../doctors/age.php"; ?><?php echo $c; ?></li>
<?php } ?> <?php }  ?>
</ol>
</nav>
<form action="discharge.php" method="GET">
<span><?php
include "../pharmacy/patient_search.php";
?>
<input type="hidden" name="response" value="0"> <button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>     
</form> 
<?php
$search=$_GET['search'];
if ($search==0) {
//do nothing
}
if ($search!=0) { 
$patient=$search;   

?>
<p></p>

<div class="container">        
<div class="col-sm-4" >      
<form action="savedischarge.php" method="POST">
<label>write nursing discharge notes</label>
<input type="hidden" name="pn" value="<?php echo $patient; ?>">
<textarea style="width: 30em;height:15em;" name="notes"></textarea></br>
</div></div>
<button class="btn btn-success" style="width: 40%;margin-left: 3%;">submit </button>
</form>
</div>
</div>
</hr>
<?php }  ?> 

<?php
$respose=$_GET['response'];

if ($respose==6) {

?>
<div class="alert alert-danger" style="width: 70%;margin-left: 1%;"><p> no authorization from doctor to discharge the patient</p></div>

</div><?php } ?>

<script src="../pharmacy/dist/vertical-responsive-menu.min.js"></script>
</div></div></div></div>

</body>
</html>