<?php
include('../connect.php');
$paid_out =1;
$ids =$_POST["id"];
foreach ($ids as $id ) {
$served =1;
$sql ="UPDATE collection
        SET  paid_out=?
		WHERE collection_id=?";;
$q = $db->prepare($sql);
$q->execute(array($paid_out,$id));
}
header("location:professional.php?success=1");
?>