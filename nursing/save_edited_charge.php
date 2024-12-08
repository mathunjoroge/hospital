<?php
session_start(); 
include('../connect.php');
    $id=$_POST['collection_id'];
    $amounts=$_POST['amount'];
     foreach (array_combine($id, $amounts) as $id => $amount) {
$sql = "UPDATE collection
        SET  amount=?
		WHERE collection_id=?";
$q = $db->prepare($sql);
$q->execute(array($amount,$id));
}
$token=mt_rand(10000000, 99999999);

if (isset($_POST['edit'])) {
header("location:theatercharge.php?token=$token&success=1");
} 
else{
header("location:procedure.php?token=$token&success=1");
}
 ?>