<?php
session_start(); 
include('../connect.php');
$id=$_POST['id'];
$reset =3;
$sql = "UPDATE lab
SET  served=?
WHERE id=?";
$q = $db->prepare($sql);
$q->execute(array($reset,$id));
// Redirect the user back to the previous page
header('Location: '.$_SERVER['HTTP_REFERER']);
exit;
?>