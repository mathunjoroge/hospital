<?php 
require_once('../main/auth.php');
include ('../connect.php');
//check how many pharmacy products are running low
$result = $db->prepare("SELECT * FROM drugs WHERE pharm_qty<=reorder_ph");
        $result->execute();
        $lowstock = $result->rowcount();
       
        //check how many store products are running low
$result = $db->prepare("SELECT * FROM drugs WHERE quantity<=reorder_st");
        $result->execute();
        $lowstore = $result->rowcount();
        $result = $db->prepare("SELECT * FROM orders");
        $result->execute();
        $rowcountt = $result->rowcount();
        $rowcount = $rowcountt+1;
        $code=$rowcount;      
        
 ?>
 <!DOCTYPE html>
<html>
<title>pharmacy stock take</title>
<?php
  include "../header.php";
  ?>
</head>
<body>
    <!--  setting selling price not less than buying price -->
   
<script>
  function minValue(){
    let buying_price = document.getElementById("buying_price").value;
    let selling_price = document.getElementById("selling_price").value;
    if (selling_price < buying_price) {
      document.getElementById("field2").value = buying_price;
      alert("Value for selling_price cannot be lower than buying_price!");
    }
  }
  document.getElementById("selling_price").onchange = minValue;
</script>

  <header class="header clearfix" style="background-color: #3786d6;">

    <?php include('../main/nav.php'); 
    include('../connect.php');?>
   
  </header><?php
  include('side.php'); ?>  
      <div class="jumbotron" style="background: #95CAFC;">
         <body onLoad="document.getElementById('country').focus();">
      <div class="container" id="results" >
        <h3>pharmacy stocks</h3><span>
            <?php if (isset($_GET['message'])) {
    ?>
 <div class="container">
     <p class="text text-success">drugs have been updated</p>
 </div>
 <?php     // code...
 } ?>
       <input type="text" id="search" onkeyup="myFunction()" placeholder="filter any column.." title="Type in a drug">
       <form action="save_s_take.php" method="POST">
     <table class="table table-bordered" id="products_table" >
<thead class="bg-primary">
<tr>
<th>generic name</th>
<th>brand name</th>
<th>qty in pharmacy</th>
<th>buying price</th>
<th>selling price</th>
<th>insurance selling price</th>
</tr>
</thead>
<?php
        $result = $db->prepare("SELECT drug_id,price AS buying_price,  generic_name, brand_name,price*ins_mark_up AS ins_price ,price*mark_up AS selling_price, quantity,pharm_qty FROM drugs");
  $result->execute();
  for($i=0; $row = $result->fetch(); $i++){     
      $drug = $row['generic_name'];
      $brand = $row['brand_name'];
      $drug_id= $row['drug_id'];
      $qty= $row['quantity'];
      $buying_price= $row['buying_price'];
      $selling_price= round($row['selling_price']);
      $qtyp= $row['pharm_qty'];
      $tqty= $qty+$qtyp;
      $ins_price=$row["ins_price"];
         ?>
<tbody>
<tr>
    
<td><?php echo $drug; ?></td>
<td><?php echo $brand; ?></td>
<input type="hidden" name="drug_id[]" value="<?php echo $drug_id; ?>" >
<td ><input type="number" name="qtyp[]" value="<?php echo $qtyp; ?>" contenteditable="true"></td>
<td ><input type="number" name="buying_price[]" id="buying_price" value="<?php echo round($buying_price); ?>" contenteditable="true"></td>
<td ><input type="number" name="selling_price[]" id="selling_price" value="<?php echo round($selling_price); ?>" contenteditable="true"></td>
<td ><input type="number" name="ins_price[]" id="selling_price" value="<?php  echo round($ins_price); ?>" contenteditable="true"></td>
<?php }?>
</tr>
<tr> 
</tbody>
</table>
 <button class="btn btn-success"><i class="icon icon-save icon-large"></i>save</button></span> 
</form>

 </br>   

</div> </div>      
      
</div>
<script>
var $rows = $('#products_table tbody tr');
$('#search').keyup(function() {
    
    var val = '^(?=.*\\b' + $.trim($(this).val()).split(/\s+/).join('\\b)(?=.*\\b') + ').*$',
        reg = RegExp(val, 'i'),
        text;
    
    $rows.show().filter(function() {
        text = $(this).text().replace(/\s+/g, ' ');
        return !reg.test(text);
    }).hide();
});
</script>

 
</div></div></div></div></div></div></div></div>

</body>
</html>