<!DOCTYPE html>
<html>
<head>
<style>
table {
width: 100%;
border-collapse: collapse;
}

table, td, th {
border: 1px solid black;
padding: 5px;
}

th {text-align: left;}
</style>
</head>
<body>

<?php
$a = intval($_GET['q']);

include('../connect.php');
$result = $db->prepare("SELECT drug_id AS drug,pharm_qty, price*mark_up AS price FROM drugs WHERE drug_id =:a");
$result->BindParam(':a', $a);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$price=$row['price'];
$drug=$row['drug'];
$qty=$row['pharm_qty'];

?>
<?php }?>
<span><input type="hidden" name="id" value="<?php echo $drug; ?>">
<td width="4" ><input type="number"  value="<?php echo $price; ?>" style="width:10rem;" ></td>
<td><input type="number"  value="<?php echo $qty; ?>"  style="width:28rem;" contenteneditable="false" required></td>
<td width="4"><input name="qty" style="width:7rem;" type="number" min="1" max="<?php echo $qty;  ?>" placeholder="qty" required/></td>

</tr>
</tbody>
</table>
</body>
</html>