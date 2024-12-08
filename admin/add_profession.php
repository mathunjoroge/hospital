<?php
include('../connect.php');
  ?>
 <center><h3>Add a profession</h3></center>
</hr>
  <form action="save_profession.php" method="POST">
	<table class="thead-dark" style="width:80%;" >
<thead>
<tr>
<th>name</th>
</tr>
</thead>
<tbody>
<tr>
<td><input type="text" name="name" style="outline: none;" required/></td>
</tr>
</tbody>
</table>
<p>&nbsp;</p>
<button class="btn btn-success" style="width: 50%;">save</button>
</form>