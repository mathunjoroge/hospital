<?php
include('../connect.php');
  ?>
 <center><h3>Add a professional</h3></center>
</hr>
  <form action="save_professional.php" method="POST">
	<table class="thead-dark" style="width:80%;" >
<thead>
<tr>
<th>name</th>
<th>profession</th>
</tr>
</thead>
<tbody>
<tr>
<td><input type="text" name="name" style="outline: none;" required/></td>
<td><select id="profession" name="profession">
  <option></option>
  <option disabled>select profession</option>
<?php 
        $result = $db->prepare("SELECT * FROM professions");
        $result->execute();
        for($i=0; $row = $result->fetch(); $i++){                   
        ?> 
        <option value="<?php echo $row['profession_id']; ?>"><?php echo $row['name']; ?></option>
        <?php } ?>     
</select></td>
</tr>
</tbody>
</table>
<p>&nbsp;</p>
<button class="btn btn-success" style="width: 100%;">save</button>
</form>