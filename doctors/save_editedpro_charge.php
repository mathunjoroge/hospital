<?php
session_start(); 
include('../connect.php');
    $id=$_POST['collection_id'];
   
    $amounts=$_POST['amount'];
     foreach (array_combine($id, $amounts) as $id => $amount) {
          $done_by=$_POST['done_by'];
$sql = "UPDATE collection
        SET  amount=?,
        done_by=?
		WHERE collection_id=?";
$q = $db->prepare($sql);
$q->execute(array($amount,$done_by,$id));
}
$token=mt_rand(10000000, 99999999);

     header("location:pro.php?token=$token&success=1");


 ?>