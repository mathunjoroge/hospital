<?php
session_start(); 
include('../connect.php');
$a=$_POST['qty'];
$id=$_POST['id'];
$token=$_POST['token'];
$diff=$d-$a;
$sql = "UPDATE dispensed_drugs
SET  quantity=?
WHERE dispense_id=?";
$q = $db->prepare($sql);
$q->execute(array($a,$id));

header("location: walkin.php?receipt=$token") ?>