<?php
include('../connect.php');
  ?><center><h3>Add product</h3></center>
</hr>
  <form action="saveproduct.php" method="POST">
<div class="card">
	<div class="row">
		<div class="col-sm-6">
			<label>generic name</label></br>
			<input type="text" name="gen" style="outline: none;"></br>
			<label>brand name</label></br>
			<input type="text" name="brand"></br>
			<label>category</label></br>
			<select name="category">
    <option value="" disabled="">-- Select category--</option><?php 
$result = $db->prepare("SELECT * FROM drug_category");
        $result->execute();
        for($i=0; $row = $result->fetch(); $i++){
           echo "<option value=".$row['name'].">".$row['name']."</option>"; }
         
        
        ?>      
</select></br>
			<label>quantity</label></br>
			<input type="text" name="qty"></td>
		</div>
		<div class="col-sm-6">			
			<label>buying price</label></br>
			<input type="text" name="price"></br>
			<label>selling price</label></br>
			<input type="number" name="selling"></br>
			<label>insurance price</label></br>
			<input type="number" name="ins_mark_up" ></br>
			<label>reoder pharm</label></br>
			<input type="number" name="reorderph" ></br>					
		</div>
	</div>	
</div>
<p>&nbsp;</p>
<button class="btn btn-success" style="width: 100%;">save</button>
</form>