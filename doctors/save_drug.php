<?php
session_start(); 
include('../connect.php');
$generic =$_POST['generic'];
$brand =$_POST['brand'];

$sql = "INSERT INTO meds (ActiveIngredient,DrugName) VALUES (:a,:b)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$generic,':b'=>$brand));
header("location:index.php?message=1");
?>