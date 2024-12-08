<?php
session_start();

include('../connect.php');
$a=$_GET['id'];
$e=$_GET['qty'];
$token=$_GET['token'];
$sql = "DELETE FROM `dispensed_drugs` WHERE `dispensed_drugs`.`dispense_id` = $a";
$q = $db->prepare($sql);
$q->execute();
// edit qty if the patient is admitted
 
	header("location: walkin.php?receipt=$token");

?>