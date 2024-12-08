<?php
session_start(); 
include('../connect.php');
$a=$_GET['code'];
$b=1;
$token=rand();
$sql = "UPDATE prescribed_meds
        SET  dispensed=?
		WHERE patient=?";
$q = $db->prepare($sql);
$q->execute(array($b,$a));
//set patient available at cashier
$sql = "UPDATE patients
        SET  has_bill=?
		WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($b,$a));
//redirect to homepage
 header("location: index.php?token=$token&search= &response=1"); 
 ?>