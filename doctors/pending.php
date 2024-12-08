<?php 
require_once('../main/auth.php');
include ('../connect.php');
 ?>
<!DOCTYPE html>
<html>
<title>peding lab results</title>
<?php
include "../header.php";
?>
</head>

<body>
<body>
  <header class="header clearfix" style="background-color: #3786d6;">
    
    <?php include('../main/nav.php'); ?>   
  </header><?php include('side.php'); ?>
  <div class="content-wrapper">
      <div class="jumbotron" style="background: #95CAFC;">
      <p>&nbsp;</p>
<div class="container" >   <nav aria-label="breadcrumb" style="width: 90%;">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
    <li class="breadcrumb-item active" aria-current="page">lab patients</li>
    <li class="breadcrumb-item active" aria-current="page">results pending</li>
  </ol>
</nav> 
     <table class="resultstable" >
<thead>
<tr>
<th>patient name</th>
<th>sex</th>
<th>age</th>

</tr>
</thead>
<?php 
$served=1;
$result = $db->prepare("SELECT name,age,sex,opno FROM patients RIGHT OUTER JOIN lab ON lab.opn=patients.opno WHERE lab.served=:served GROUP BY opno");
$result->bindParam(':served',$served);
  $result->execute();
  for($i=0; $row = $result->fetch(); $i++){
     
      $name = $row['name'];
      $b = $row['age'];
      $c = $row['sex'];
      $number= $row['opno'];

         ?>
        
<tbody>
<tr>
<td><a rel="facebox" href="labview.php?id=<?php echo $number ?>&name=<?php echo $name ?>"><?php echo $name; ?></a></td>
<td><?php echo $c; ?></td>
<td>  <?php include "age.php";?></td>

<?php }?>
</tr>

</tbody>
</table>
 </br>
<script src="../pharmacy/dist/vertical-responsive-menu.min.js"></script>
</div></div></div></div></div></div></div></div>

</body>
</html>