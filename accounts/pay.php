<?php include ('../connect.php');
$patient=$_GET['id'];
$med_amount=$_GET['med_amount'];
$lab=$_GET['lab'];
$clinic=$_GET['clinic'];
$fees=$_GET['fees'];
$total=$_GET['total'];
$days=$_GET['du'];
$result = $db->prepare("SELECT  GROUP_CONCAT(drugs.drug_id) as drug_id,GROUP_CONCAT(dispensed_drugs.quantity) as quantity FROM dispensed_drugs RIGHT OUTER JOIN drugs ON drugs.drug_id=dispensed_drugs.drug_id WHERE patient='$patient' AND cashed_by=''");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$drug_id= $row['drug_id'];
$quantity= $row['quantity'];    

?>
<script type="text/javascript">
$("#payment").change(function(){
if($(this).val() == 2){
$("#mpesa").show();
$("#mpe").show();
$("#mpes").show();
$("#insurance").hide();
$("#ins").hide();
$("#insu").hide();
$("#ban").hide();
$("#bank").hide();
$("#banki").hide();
$("#insurance").value="";
}
else if($(this).val() == 3){
$("#insurance").show();
$("#ins").show();
$("#insu").show();
$("#mpesa").hide();
$("#mpe").hide();
$("#mpes").hide();
$("#ban").hide();
$("#bank").hide();
$("#banki").hide();
}
else if($(this).val() == 4){
$("#ban").show();
$("#bank").show();
$("#banki").show();
$("#insurance").hide();
$("#ins").hide();
$("#insu").hide();
$("#mpesa").hide();
$("#mpe").hide();
$("#mpes").hide();
$("#insurance").value="";
}
else{
$("#ban").hide();
$("#bank").hide();
$("#banki").hide();
$("#mpesa").hide();
$("#mpe").hide();
$("#mpes").hide();
$("#insurance").hide();
$("#ins").hide();
$("#insu").hide();

}

});
</script>
<h5 align="center">cash payment</h5>
<form action="savepay.php" method="POST">
<table class="resultstable" >
<thead>
<tr>
<th>total amount</th>
<th>cash tendered</th>
<th>payment mode</th>
<th id="mpes">Mpesa</th>
<th id="insu">Insurance</th>
<th id="ban">bank</th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td><?php echo $total; ?></td>
<td><input style="outline: none;width: 7em;" type="number" name="tendered" >
<td><select id="payment" name="payment_mode" title="select payment mode" placeholder="select payment mode" required>
<option value="1">cash</option>
<option value="2">Mpesa</option>
<option value="3">Insurance</option> 
<option value="4">bank</option> 
</select></td>
<td id="mpe"><input name="mobile" placeholder="enter mpesa code" id="mpesa"></td>
<td id="ban"><input name="bank" placeholder="enter bank conf code" id="banki"></td>
<td id="ins"><select id="insurance" name="insurance">
<option></option>
<option>Choose insurance company</option>
<?php 
$result = $db->prepare("SELECT * FROM insurance_companies");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){                   
?> 
<option value="<?php echo $row['company_id']; ?>"><?php echo $row['name']; ?></option>
<?php } ?>     
</select> </td>
<input type="hidden" name="id" value="<?php echo $_GET['id']; ?>">
<input type="hidden" name="med_amount" value="<?php echo $med_amount; ?>">
<input type="hidden" name="clinic" value="<?php echo $clinic; ?>">
<input type="hidden" name="fees" value="<?php echo $fees; ?>">
<input type="hidden" name="lab" value="<?php echo $lab; ?>">
<input type="hidden" name="amount" value="<?php echo $total; ?>">
<input type="hidden" name="ward" value="<?php echo $_GET['wards_income']; ?>">
<input type="hidden" name="updated" value="<?php echo date("d-m-Y h:i:s"); ?>">
<input type="hidden" name="drug_id[]" value="<?php echo $drug_id; ?>">
<input type="hidden" name="quantity[]" value="<?php echo $quantity; ?>">
<input type="hidden" name="du" value="<?php echo $days; ?>">
</tbody>
</table></br>
<button class="btn btn-success btn-large" style="width: 100%;">save</button>  
</form><?php } ?>
