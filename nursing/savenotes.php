<?php
session_start(); 
include('../connect.php');
$a = date('Y-m-d H:i:s');
$j = $_GET['pn'];
$b = $_GET['notes'];
$doctor =$_SESSION['SESS_FIRST_NAME'];
$sql = "INSERT INTO patient_notes (created_at,patient,notes,posted_by) VALUES (:a,:b,:c,:d)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$a,':b'=>$j,':c'=>$b,':d'=>$doctor));

 $has_bill =1;
$sql = "UPDATE patients
        SET  has_bill=?
		WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($has_bill,$j)); 

header("location: notes.php?search=$j&response=1"); 
?>

    