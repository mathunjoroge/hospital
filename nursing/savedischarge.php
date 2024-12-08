<?php
session_start(); 
include('../connect.php');
$a = $_POST['pn'];
$b = $_POST['notes']; 
$d =$_SESSION['SESS_FIRST_NAME'];
//
$sql = "UPDATE discharge_summary
SET  nursing_notes=?,
nurse=?
WHERE patient=?";
$q = $db->prepare($sql);
$q->execute(array($b,$d,$a));

header("location: discharge.php?search=0&response=1");

?>