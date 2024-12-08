<?php include('../connect.php');
$a=$_POST['name'];
$b=$_POST['address'];
$c=$_POST['phone'];
$d=$_POST['slogan'];
$e=$_POST['email'];
$f=$_POST['p_from'];
$zone=$_POST['zone'];
$sql = "INSERT INTO settings (name,address,phone,slogan,email,fda_user) VALUES (:a,:b,:c,:d,:e,:f)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$a,':b'=>$b,':c'=>$c,':d'=>$d,':e'=>$e,':f'=>$f));

//setting user timezone
$path_to_file = '../.htaccess';
$file_contents = file_get_contents($path_to_file);
$file_contents = str_replace("Africa/Cairo",$zone,$file_contents);
file_put_contents($path_to_file,$file_contents);
//redirecting

header("location: index.php");


?>