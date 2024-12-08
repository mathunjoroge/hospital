<?php
session_start(); 
include('../connect.php');
    $a=$_POST['id'];
    $b=$_POST['name'];
    $c=$_POST['amount'];
    $ins_cost=$_POST['ins_cost'];     
       
$sql = "UPDATE lab_tests
        SET  name=?,
             cost=?,
             ins_cost=?              
		WHERE id=?";
$q = $db->prepare($sql);
$q->execute(array($b,$c,$ins_cost,$a));
header("location: fees.php?response=2");

 ?>
 