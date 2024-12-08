<?php 
require_once('../main/auth.php');
 ?>
 <!DOCTYPE html>
<html>
<title>patient notes</title>
<head>
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
  <div class="jumbotron" style="width:auto;background: #95CAFC;">
  <nav aria-label="breadcrumb" style="width: 90%;">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
    <li class="breadcrumb-item active" aria-current="page">patient</li>
    
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
        $d=$row['opno'];
     
     ?>
     <li class="breadcrumb-item active" aria-current="page"><?php echo $a;  ?>
     <li class="breadcrumb-item active" aria-current="page">patient notes</li> <?php
include '../doctors/age.php'; 
?>

<?php } ?>
</nav>  
   
<form action="notes.php?&response=0" method="GET">
  <?php
  include "../pharmacy/patient_search.php";
  ?><input class="form-control" type="hidden" name="response" value="0"> <button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span> </form>
    <p>&nbsp;</p>
    <?php
    $search=$_GET['search'];
    $nothing="";
    if ($search!=$nothing) {
       # code...
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
 ?>
<form action="savenotes.php" method="GET ">
        <input type="hidden" name="pn" value="<?php echo $_GET['search']; ?> ">
        
      <label>add nursing notes and cardex</label></br> 
      <textarea name="notes" style="width: 50%;height:10em;"></textarea></br>
    </br>
<button class="btn btn-success btn-large" style="width: 65%;">save</button></form><?php } ?>
<?php
 ?> 
</div>
</div>
</div>

<script src="../pharmacy/dist/vertical-responsive-menu.min.js"></script>

</body>
</html>

