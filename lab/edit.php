<?php 
require_once('../main/auth.php');
include ('../connect.php');
 ?>
<!DOCTYPE html>
<html>
<title>lab</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=false">
<meta http-equiv="refresh" content="30"/>
  <link href='../pharmacy/src/vendor/normalize.css/normalize.css' rel='stylesheet'>
  <link href='../pharmacy/src/vendor/fontawesome/css/font-awesome.min.css' rel='stylesheet'>
  <link href="../pharmacy/dist/vertical-responsive-menu.min.css" rel="stylesheet">
  <link href="../pharmacy/demo.css" rel="stylesheet">
  <link rel="stylesheet" href="../css/bootstrap.min.css">
  <link rel="stylesheet" href="dist/css/bootstrap-select.css">
  <script src="../js/jquery.min.js"></script>
  <script src="../js/bootstrap.min.js"></script>
  <script src="dist/js/bootstrap-select.js"></script>
   
</head>

<body>
<body>
  <header class="header clearfix" style="background-color: #3786d6;">
    
    <?php include('../main/nav.php'); ?>   
  </header><?php include('side.php'); ?>
  <div class="content-wrapper">
      <div class="jumbotron" style="background: #95CAFC;">
         <body onLoad="document.getElementById('patient').focus();">

 
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
      <p>&nbsp;</p>
  
<div class="container" >   <nav aria-label="breadcrumb" style="width: 90%;">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
    <li class="breadcrumb-item active" aria-current="page">lab patients</li>
    <li class="breadcrumb-item active" aria-current="page">waiting list</li>
    <li class="breadcrumb-item"><a href="edit.php?search= &response=0">edit results</a></li>
  </ol>
</nav> 
<?php if (isset($_REQUEST["failed"])) {
  # code...
 ?>
 <b class="alert-danger"> you did not tick all the comments. please repeat and post</b>
<?php } ?>
<?php if (isset($_REQUEST["success"])) {
  # code...
 ?>
 <b class="alert-success"> data posted successifully</b>
<?php } ?>
     <table class="table table-bordered" >
<thead>
<tr>
<th>patient name</th>
<th>sex</th>
<th>age</th>

</tr>
</thead>
<?php
$a =1;
$result = $db->prepare("SELECT DISTINCT(opno),age,sex,name FROM patients RIGHT OUTER JOIN lab ON lab.opn=patients.opno WHERE lab.served=:a");
$result->bindParam(':a',$a);
  $result->execute();
  for($i=0; $row = $result->fetch(); $i++){
     
      $name = $row['name'];
      $b = $row['age'];
      $c = $row['sex'];
      $number= $row['opno'];

         ?>
        
<tbody>
<tr>
<td><a  href="details.php?search=<?php echo $number ?>&response=<?php echo '0'; ?>&edit=1"><?php echo $name; ?></a></td>
<td><?php echo $c; ?></td>
<td>  <?php 
  include "../doctors/age.php"; ?></td>

<?php }?>
</tr>

</tbody>
</table>

</tr>

</tbody>
</table></div></div>
 </br>
<script src="../pharmacy/dist/vertical-responsive-menu.min.js"></script>
</div></div></div></div></div></div></div></div>

</body>
</html>