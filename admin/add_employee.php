<?php
include('../connect.php');
?>
<div class="container" style="width: 40em;">
  <center><h3>Add employee</h3></center>
<form action="save_employee.php" method="POST">
<div class="row">
<div class="col-sm-4">
<label>name</label></br>
<input type="text" name="employee_name" style="outline: none;" title="select job group" required/>
<label>job group</label></br>
<select name="jg">
<option selected disabled>--select j group--</option>
<?php
$result = $db->prepare("SELECT*  FROM job_groups");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){     
$jg_id= $row['jg_id'];
$jg_name= $row['jg_name']; ?>
<option value="<?php echo $jg_id; ?>"><?php echo $jg_name; ?></option><?php } ?></select>
</div>
<div class="col-sm-4">
<label>id number</label></br>
<input type="text" name="id_number" style="outline: none;" required/> 
<label>NHIF NO</label></br>
<input type="text" name="nhif" style="outline: none;" required/>
<label>NSSF NO</label></br>
<input type="text" name="nssf" style="outline: none;" required/>
</div>
<div class="col-sm-4">
<label>bank</label></br>
<input type="text" name="bank" style="outline: none;" required/>
<label>account no</label></br>
<input type="text" name="acc" style="outline: none;" required/>
</div>
</div>
<button class="btn btn-success" style="width: 100%;">save</button>
</form>
</div>
