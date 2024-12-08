<?php
session_start(); 
include '../connect.php';
$cashier=$_SESSION['SESS_FIRST_NAME'];
$receipt=$_GET['receipt'];
$total=$_GET['total'];
$result = $db->prepare("SELECT*  FROM  dispensed_drugs   WHERE  token=:b");
$result->BindParam(':b', $receipt);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
 $drug_id=$row['drug_id'];
 $quantity=$row['quantity'];
 //update the drugs to minus the quantities
 $sql = "UPDATE drugs
        SET  pharm_qty=pharm_qty-?
		WHERE drug_id=?";
$q = $db->prepare($sql);
$q->execute(array($quantity, $drug_id));

//save the total

$sql = "INSERT INTO cash (amount,cashtendered,tendered_by,receipt_no) VALUES (:a,:b,:c,:g)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$total,':b'=>$total,':c'=> $cashier,':g'=> $receipt));
header("location: receipt.php?receipt=$receipt&mode=1"); 



}

?>