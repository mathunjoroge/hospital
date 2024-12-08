<?php 
require_once('../main/auth.php');
include('../connect.php');
$token=$_GET['token'];
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

      <div class="container" ><label>select medicines to order</label></br>
      <div class="row">
          <div class="col-md-6">
        <form action="saveorder.php" method="POST">
        <input type="hidden" name="pn" value="<?php echo $_GET['req']; ?>">
      
      <span><select id="medicine" name="med" class="selectpicker form-control" data-live-search="true" title="Please select a medicine..." >
<?php 
           include ('../connect.php');
          $result = $db->prepare("SELECT * FROM drugs");
                  $result->execute();
                  for($i=0; $row = $result->fetch(); $i++){
                     echo "<option value=".$row['drug_id'].">".$row['generic_name']."-".$row['brand_name']."</option>";
                   }
                  
                  ?> 
</select>
       </div>
       <div class="col-md-6">
      <input name="qty" style="" class="form-label" type="number" min="1" title="enter qty">&nbsp;<button class="btn btn-success btn-large">add</button></form></span>
         </div></div>
      
      <div class="container" id="results" > <label>selected meds</label></br> 
     <table class="resultstable" >
<thead>
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
       $patient=$_GET['req'];
        $result = $db->prepare("SELECT drugs.drug_id,generic_name,brand_name,price,dispense_id,orders.quantity FROM orders RIGHT OUTER JOIN drugs ON drugs.drug_id=orders.drug_id WHERE patient='$patient' AND cashed_by=''");
  $result->execute();
  for($i=0; $row = $result->fetch(); $i++){
     
      $drug = $row['generic_name'];
      $brand = $row['brand_name'];
      $price= $row['price'];
      $qty= $row['quantity'];
         ?>
<tbody>
<tr>
<td><?php echo $drug; ?></td>
<td><?php echo $brand; ?></td>
<td ><?php echo $qty; ?></td>
<td><?php echo $price; ?></td>
<td ><?php  echo $qty*$price; ?></td>
<td><a href="delete_order.php?id=<?php echo $row['dispense_id']; ?>"><button class="btn btn-danger" style="height: 5px;" title="Click to Delete"></td><?php }?>
</tr>
<tr> <?php $patient=$_GET['req'];
        $result = $db->prepare("SELECT sum(price*orders.quantity) as total FROM orders RIGHT OUTER JOIN drugs ON drugs.drug_id =orders.drug_id WHERE patient='$patient' AND cashed_by=''");
  $result->execute();
  for($i=0; $row = $result->fetch(); $i++){ ?>
      <th> </th>
      <th>  </th>
      <th>  </th>
      <th>  </th>
      <td> Total Amount: </td>
      
    </tr>
      <tr>
        <th colspan="4"><strong style="font-size: 12px; color: #222222;">Total:</strong></th>
        <td colspan="1" id="myvalue"><strong style="font-size: 12px; color: #222222;"> <?php echo $row['total']; ?> </td><?php } ?>
</tbody>
</table>
 </br>
 <a href="index.php?search=%20&response=0">
 <button class="btn btn-success btn-large" style="width: 100%;">save</button></a>
      

</div> </div>
      
      
</div>

      
<script>
  $(document).ready(function () {
    var mySelect = $('#first-disabled2');

    $('#special').on('click', function () {
      mySelect.find('option:selected').prop('disabled', true);
      mySelect.selectpicker('refresh');
    });

    $('#special2').on('click', function () {
      mySelect.find('option:disabled').prop('disabled', false);
      mySelect.selectpicker('refresh');
    });

    $('#basic2').selectpicker({
      livereq: true,
      maxOptions: 1
    });
  });
</script>
</div></div></div></div></div></div></div></div>

</body>
</html>