<?php
session_start(); 
include('../connect.php');
// set the lab tests to approved
$lab_id = $_POST['lab_id'];
foreach ($lab_id as $id ) {
$served =1;
$sql ="UPDATE lab
SET  served=?
WHERE id=?";
$q = $db->prepare($sql);
$q->execute(array($served,$id));
}
header("location:lab.php?response=1&search=0&success=1");
?>

