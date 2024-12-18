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
  <header class="header clearfix" style="background-color: #3786d6;;">
    

    </button>
    <?php include('../main/nav.php'); ?>
   
  </header><?php include('side.php'); ?>
  <div class="content-wrapper">  
  <div class="jumbotron" style="width:auto;background: #95CAFC;">
       <?php
    if (isset($_GET['success'])) {

    ?>
    <script>
        setTimeout(function() {
    $('#alert').fadeOut('fast');
}, 1000); 
    </script>
    <div id="alert">
        <p class="alert alert-success">procedure has been saved</p>
    </div>
    <?php } ?>
  <nav aria-label="breadcrumb" style="width: 90%;">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
    <li class="breadcrumb-item active" aria-current="page">patient</li>
    <li class="breadcrumb-item active" aria-current="page">procedure fees</li>
    
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
     ?>
     <li class="breadcrumb-item active" aria-current="page"><?php echo $a; ?>:  &nbsp;<?php  echo $c; ?>,  <?php include '../doctors/age.php'; ?></li><?php }} ?>
</nav>  
  <form action="procedure.php?&response=0" method="GET">
  <span><?php
  include "../pharmacy/patient_search.php";
  ?>
  <input type="hidden" name="response" value="0"> <button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>     
      <div class="suggestionsBox" id="suggestions" style="display: none;">
        <div class="suggestionList" id="suggestionsList"> &nbsp; </div>

</div></form>
    <p>&nbsp;</p>
      <?php 
      if (isset($_GET['search'])) {
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
  <div class="container-fluid">
       
      <h3>add procedure charges</h3>
      <form action="save_nursing_charge.php" method="POST">        
     <table class="table">
        <tr>
          <th>service</th>
          <th>amount</th>
          <th>add</th>
        </tr>
        <tr>
          <?php
        $result = $db->prepare("SELECT*  FROM  fees WHERE is_nursing=1");
        $result->execute();
       for($i=0; $row = $result->fetch(); $i++){
      $id = $row['fees_id'];
      $fee = $row['fees_name']; 
      $amount =$row['amount'];  
     
  
         ?>
         <td><?php echo $fee; ?></td>
         <td><input name="amount[]" value="<?php echo $amount; ?>" type="number" contenteditable="true" required/></td>
         <input type="hidden" name="patient" value="<?php echo $_GET['search']; ?>">
         <td><input type="checkbox" name="charge[]" value="<?php echo $id; ?>"></td>
        </tr>
        <?php } ?>
</table>
<button class="btn btn-success btn-large" style="width: 70%;">save charge</button></form><?php }   ?> </form><?php }   ?>
</div>

        
      </div>
      </div></div></div>


<script src="../pharmacy/dist/vertical-responsive-menu.min.js"></script>

</body>
</html>