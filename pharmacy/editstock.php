<?php
        include('../connect.php');
        $id=$_GET['id'];
        $result = $db->prepare("SELECT* FROM drugs WHERE  drug_id=:c");
        $result->bindParam(':c',$id);
        $result->execute();
  for($i=0; $row = $result->fetch(); $i++){
     
      $a = $row['generic_name'];
      $b = $row['brand_name'];
      $c= $row['category'];
      $d= $row['quantity'];
      $e= $row['pharm_qty'];
      $f= $row['price'];
      $gg = $row['mark_up'];
      $g =$f*$gg;
      $h= $row['reorder_ph'];
      $ins_mark_up= $row['ins_mark_up'];
      
     $insprice=($ins_mark_up*$f);
  }
         ?>
         <h5 align="center">edit product</h5>
<form action="saveeditstock.php" method="POST">
	<table class="resulttable" style="width:80%;" >
<thead>
<tr>
<th>generic name</th>
<th>brand name</th>
<th>category</th>
<th>qty ph</th>


</tr>
</thead>

<tbody>
<tr>
<td><input type="text" name="a" value="<?php echo $a; ?>" ></td>
<td><input type="text" name="b" value="<?php echo $b; ?>"></td>
<td><select name="c">
    <option value="" disabled="">-- Select category--</option><?php 
$result = $db->prepare("SELECT * FROM drug_category");
        $result->execute();
        for($i=0; $row = $result->fetch(); $i++){
           echo "<option value=".$row['id'].">".$row['name']."</option>"; }
         
        
        ?>      
</select></td>
<td ><input type="text" name="e" value="<?php echo $e; ?>"></td>
</tr>
<tr> 
</tbody>
</table>
<p>&nbsp;</p>
<table class="thead-dark" style="width:80%;" >
<thead>
<tr>

<th>buying price</th>
<th>selling price</th>
<th>reoder pharm</th>
<th>insurance price</th>

</tr>
</thead>

<tbody>
<tr>

<td><input type="text" name="f" value="<?php echo $f; ?>"></td>
<td><input type="text" name="g" value="<?php echo $g; ?>"></td>
<td><input type="text" name="h" value="<?php echo $h; ?>" ></td>
<td><input type="text" name="j" value="<?php echo $insprice; ?>"></td>
<td><input type="hidden" name="k" value="<?php echo $id; ?>"></td>

</tr>
<tr> 
</tbody>
</table>
<p>&nbsp;</p>
<button class="btn btn-success" style="width: 100%;">save</button>
</form>
	
</form>