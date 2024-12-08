<?php 
require_once('../main/auth.php');
include ('../connect.php');

 ?>
 
 <!DOCTYPE html>
<html>

  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">

  
  <link href='src/vendor/normalize.css/normalize.css' rel='stylesheet'>
  <link href='src/vendor/fontawesome/css/font-awesome.min.css' rel='stylesheet'>
  <link href="dist/vertical-responsive-menu.min.css" rel="stylesheet">
  <link href="demo.css" rel="stylesheet">
  <link rel="stylesheet" href="../css/bootstrap.min.css">
  <link rel="stylesheet" href="dist/css/bootstrap-select.css">
  <script src="../js/jquery.min.js"></script>
  <script src="../js/bootstrap.min.js"></script>
  <script src="dist/js/bootstrap-select.js"></script>
  <link href="../src/facebox.css" media="screen" rel="stylesheet" type="text/css" />
<script src="../src/facebox.js" type="text/javascript"></script>

<script type="text/javascript">
  function printContent(el){
var restorepage = $('body').html();
var printcontent = $('#' + el).clone();
$('body').empty().html(printcontent);
window.print();
$('body').html(restorepage);
}
</script>
<title>invoice number=<?php echo $_GET['invoice']; ?></title>
</head><header class="header clearfix" style="background-color: #3786d6;">
    
    <?php include('../main/nav.php'); ?>   
  </header><?php include('side.php'); ?>
  <div class="content-wrapper"> 
      <div class="jumbotron" style="background: #95CAFC;">
         <body onLoad="document.getElementById('country').focus();"> 
       <style>#amount {
  text-align: right;
}
#description {
  text-align: left;
}
#invoice-POS{
  box-shadow: 0 0 1in -0.25in rgba(0, 0, 0, 0.5);
  padding:2mm;
  margin: 0 auto;
  width: 44mm;
  background: #FFF;
  
  
::selection {background: #f31544; color: #FFF;}
::moz-selection {background: #f31544; color: #FFF;}
h1{
  font-size: 1.5em;
  color: #222;
}
h2{font-size: .9em;}
h3{
  font-size: 1.2em;
  font-weight: 300;
  line-height: 2em;
}
p{
  font-size: .7em;
  color: #666;
  line-height: 1.2em;
}
 
#top, #mid,#bot{ /* Targets all id with 'col-' */
  border-bottom: 1px solid #EEE;
}

#top{min-height: 100px;}
#mid{min-height: 80px;} 
#bot{ min-height: 50px;}

#top .logo{
  //float: left;
	height: 60px;
	width: 60px;
	background: url(http://michaeltruong.ca/images/logo1.png) no-repeat;
	background-size: 60px 60px;
}
.clientlogo{
  float: left;
	height: 60px;
	width: 60px;
	background: url(http://michaeltruong.ca/images/client.jpg) no-repeat;
	background-size: 60px 60px;
  border-radius: 50px;
}
.info{
  display: block;
  //float:left;
  margin-left: 0;
}
.title{
  float: right;
}
.title p{text-align: right;} 
table{
  width: 100%;
  border-collapse: collapse;
}
td{
  //padding: 5px 0 5px 15px;
  //border: 1px solid #EEE
}
.tabletitle{
  //padding: 5px;
  font-size: .5em;
  background: #EEE;
}
.service{border-bottom: 1px solid #EEE;}
.item{width: 24mm;}
.itemtext{font-size: .5em;}

#legalcopy{
  margin-top: 5mm;
}

  
  
}
</style>
     <div class="container-fluid">
<div class="container" id="print" > </br>

<div id="invoice-POS">    
<center id="top">
<div class="logo"></div>
<div class="info">  
     <div class="logo-container" style="width: 18.3em; height: 8.4em;" class="center" >
<img src="../icons/logo.jpg"  style="width: 100%; height: 100%;">
</div>
</div>
</div><!--End Info-->
</center><!--End InvoiceTop-->
</div>
<center><h4>purchases</h4></center>
<center><h4>invoice number: <?php echo $_GET["invoice"] ?></h4></center>
<table class="table table-bordered" >
<thead class="bg primary" >
<tr>
<th>generic name</th>
<th>brand name</th>
<th>quantity</th>
<th>price</th>
<th>total</th>
</tr>
</thead>
<?php
       $invoice=$_GET['req'];
        $result = $db->prepare("SELECT drugs.drug_id,generic_name,brand_name,purchases.price AS price,purchases.qty FROM purchases RIGHT OUTER JOIN drugs ON drugs.drug_id=purchases.drug_id WHERE inv='$invoice'");
  $result->execute();
  for($i=0; $row = $result->fetch(); $i++){
     
      $drug = $row['generic_name'];
      $brand = $row['brand_name'];
      $price= $row['price'];
      $qty= $row['qty'];
         ?>
<tbody>
<tr>
<td><?php echo $drug; ?></td>
<td><?php echo $brand; ?></td>
<td ><?php echo $qty; ?></td>
<td><?php echo $price; ?></td>
<td ><?php  echo $qty*$price; ?></td>
<?php }?>
</tr>
<tr> <?php $invoice=$_GET['req'];
        $result = $db->prepare("SELECT sum(purchases.price*purchases.qty) as total FROM purchases RIGHT OUTER JOIN drugs ON drugs.drug_id =purchases.drug_id WHERE inv='$invoice'");
  $result->execute();
  for($i=0; $row = $result->fetch(); $i++){ ?>
      <th> </th>
      <th>  </th>
      <th>  </th>
      <th> Total  </th>
      <td> <b><?php echo $row['total']; ?></b> </td>
      
    </tr>
      
</tbody>
</table>
</div>
 </br><button class="btn btn-success btn-large" style="width: 100%;" id="print" onclick="printContent('print');">print</button><?php } ?>    

</div> </div>     
</div>
  <script src="dist/vertical-responsive-menu.min.js"></script>
</div></div></div></div></div></div></div></div>

</body>
</html>