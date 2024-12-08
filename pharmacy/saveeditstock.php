<?php
session_start(); 
include('../connect.php');
//posting variables here
    $a=$_POST['a'];
    $b=$_POST['b'];
    $c=$_POST['c'];
    $e=$_POST['e'];
    $f=$_POST['f'];// buying price mark_up=(((selling price-buying price)*100)+1) 
    $gg=$_POST['g'];//selling price
     $j=$_POST['j'];
    $g=((($gg-$f)/$f)+1);
    $ins_mark_up=((($j-$f)/$f)+1);
    $h=$_POST['h'];
    $id=$_POST['k'];
$sql = "UPDATE drugs
        SET  generic_name=?,brand_name=?,category=?,pharm_qty=?,price=?,mark_up=?,reorder_ph=?,ins_mark_up=?
		WHERE drug_id=?";
$q = $db->prepare($sql);
$q->execute(array($a,$b,$c,$e,$f,$g,$h, $ins_mark_up,$id)); 
header("location: stocks.php") ?>