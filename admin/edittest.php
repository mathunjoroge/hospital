<h5 align="center">edit charge</h5>
<form action="saveedittest.php" method="POST">
	<table class="resultstable" >
<thead>
<tr>
<th>name</th>
<th>amount</th>
<th>insurance cost</th>
</tr>
</thead>
<tbody>
<tr>
<td ><input style="outline: none;width: 7em;" type="text" name="name" value="<?php echo $_GET['name']; ?>"required/></td>
<td ><input type="text" name="amount" value="<?php echo $_GET['amount']; ?>" required/></td >
<td ><input type="text" name="ins_cost" value="<?php echo $_GET['ins_cost']; ?>" required/></td >
<input type="hidden" name="id" value="<?php echo $_GET['id']; ?>">
</tr>
</tbody>
</table>
<button class="btn btn-success btn-large" style="width: 100%;">save</button>
	
</form>
