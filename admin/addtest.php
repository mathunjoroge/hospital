<?php
include('../connect.php');
  ?><center><h3>Add lab test</h3></center>
</hr>
  <form action="savetest.php" method="POST">
	<table class="thead-dark" style="width:80%;" >
<thead>
<tr>
<th>name</th>
<th>amount</th>
<th>insurance amount</th>
</tr>
</thead>
<tbody>
<tr>
<td><input type="text" name="test" style="outline: none;" required/></td>
<td ><input type="text" name="amount" required/></td>
<td ><input type="text" name="ins_amount" required/></td>
<input type="hidden" name="trigger" value="1" />
</tr>
</tbody>
</table>
<p>&nbsp;</p>
<button class="btn btn-success" style="width: 100%;">save</button>
</form>