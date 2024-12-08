<?php
session_start(); 
include('../connect.php');
$b = $_POST['id'];
$c = $_POST['qty'];
$token = $_POST['token'];
$d = $_SESSION['SESS_FIRST_NAME'];
$sql = "INSERT INTO dispensed_drugs (drug_id,quantity,posted_by,token) VALUES (:b,:c,:d,:e)";
$q = $db->prepare($sql);
$q->execute(array(':b'=>$b,':c'=>$c,':d'=>$d,':e'=>$token));
?>
	<script type="text/javascript">
		history.back();
	</script>

