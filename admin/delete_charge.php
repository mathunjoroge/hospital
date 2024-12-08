<?php
include('../connect.php');
$a=$_GET['id'];
$b=0;
	$sql = "UPDATE fees
        SET  status=?                       
		WHERE fees_id= ?";
$q = $db->prepare($sql);
$q->execute(array($b,$a));
	header("location: total.php?response=3");

?>